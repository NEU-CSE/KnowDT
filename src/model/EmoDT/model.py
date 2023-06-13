### TAKEN FROM https://github.com/kolloldas/torchnlp
"""rel_attention Encoder + normal Decoder"""
import os
import torch
from torch import autograd
#autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import stanza
# en_nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', download_method=None, use_gpu=True)
from src.model.common import (
    EncoderLayer,
    DecoderLayer,
    LayerNorm,
    _gen_bias_mask,
    _gen_timing_signal,
    share_embedding,
    LabelSmoothing,
    NoamOpt,
    _get_attn_subsequent_mask,
    get_input_from_batch,
    get_output_from_batch,
    top_k_top_p_filtering,
    PositionwiseFeedForward,
)
from src.model.tree_pos_enc import TreePositionalEncodings
from src.model.emo_sub_tree import get_emotion_path_mask_for_one_batch
from src.model.emo_enhance import get_emotion_words_pos, get_emo_enhance_op_order
from src.utils import config
from src.utils.constants import MAP_EMO

from sklearn.metrics import accuracy_score


class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
            self,
            embedding_size,
            hidden_size,
            num_layers,
            num_heads,
            total_key_depth,
            total_value_depth,
            # NOTE:
            total_rel_depth,
            filter_size,
            max_length=1000,
            input_dropout=0.0,
            # NOTE:
            relation_dropout=0.0,
            layer_dropout=0.0,
            attention_dropout=0.0,
            relu_dropout=0.0,
            use_mask=False,
            universal=False,
            tree_pos_enc=True,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.tree_pos_enc = tree_pos_enc
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)  # 生成位置编码

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)  # 生成位置编码
        if self.tree_pos_enc:
            self.TreePositionEncodings = TreePositionalEncodings(
                d_model=hidden_size,
                max_d_tree_param=100,
            )

        params = (
            embedding_size,
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            total_rel_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        self.relation_dropout = nn.Dropout(relation_dropout)

    def forward(self, inputs, relations, dp_map, mask, rel_mask):
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)
        rel = self.relation_dropout(relations)

        rel = self.embedding_proj(rel)

        if self.universal:
            for l in range(self.num_layers):
                if config.tree_pos_enc:
                    x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
                    x += self.TreePositionEncodings(dp_map, config.hidden_dim)
                else:
                    x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
                x += (
                    self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                )
                x = self.enc(x, rel, dp_map, mask=mask, rel_mask=rel_mask)
            y = self.layer_norm(x)
        else:
            if config.tree_pos_enc:
                x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
                x += self.TreePositionEncodings(dp_map, config.hidden_dim)
            else:
                x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, rel, dp_map, mask, rel_mask)

            y = self.layer_norm(x)
        return y

class Emotion_Encoder(nn.Module):
    """
    A Transformer Encoder module.
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        use_mask=False,
        universal=False,
        concept=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder  2
            num_heads: Number of attention heads   2
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head   40
            total_value_depth: Size of last dimension of values. Must be divisible by num_head  40
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN  50
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Emotion_Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        from src.model.common import CommonEncoderLayer
        if self.universal:
            self.enc = CommonEncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([CommonEncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.enc,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                )
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, : inputs.shape[1], :].type_as(
                        inputs.data
                    )
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
            self,
            embedding_size,
            hidden_size,
            num_layers,
            num_heads,
            total_key_depth,
            total_value_depth,
            filter_size,
            max_length=1000,
            input_dropout=0.0,
            layer_dropout=0.0,
            attention_dropout=0.0,
            relu_dropout=0.0,
            universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(
                *[DecoderLayer(*params) for l in range(num_layers)]
            )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(
            mask_trg + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)], 0
        )
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.dec,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    encoder_output,
                    decoding=True,
                )
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += (
                        self.position_signal[:, l, :]
                            .unsqueeze(1)
                            .repeat(1, inputs.shape[1], 1)
                            .type_as(inputs.data)
                    )
                    x, _, attn_dist, _ = self.dec(
                        (x, encoder_output, [], (mask_src, dec_mask))
                    )
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(
            self,
            x,
            attn_dist=None,
            enc_batch_extend_vocab=None,
            extra_zeros=None,
            temp=1,
            beam_search=False,
            attn_dist_db=None,
    ):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat(
                [enc_batch_extend_vocab.unsqueeze(1)] * x.size(1), 1
            )  ## extend for all seq
            if beam_search:
                enc_batch_extend_vocab_ = torch.cat(
                    [enc_batch_extend_vocab_[0].unsqueeze(0)] * x.size(0), 0
                )  ## extend for all seq
            logit = torch.log(
                vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_)
            )
            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class EmoDT(nn.Module):
    def __init__(
            self,
            vocab,
            decoder_number,
            model_file_path=None,
            is_eval=False,
            load_optim=False,
            is_multitask=False,
    ):
        super(EmoDT, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        # self.is_eval = is_eval
        self.multitask = is_multitask

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.encoder = Encoder(
            config.emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            total_rel_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
            tree_pos_enc=config.tree_pos_enc
        )

        ## multiple decoders
        self.emotion_embedding = nn.Linear(decoder_number, config.emb_dim)
        self.decoder = Decoder(
            config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
        )
        # NOTE: Emotion Classification (copy from CEM)
        if config.emo_enhance:
            self.emo_lin = nn.Linear(config.hidden_dim, decoder_number, bias=False)
            # for emotion enhance
            self.query_linear = nn.Linear(config.hidden_dim, config.depth, bias=False)
            self.key_linear = nn.Linear(config.hidden_dim, config.depth, bias=False)
            self.value_linear = nn.Linear(config.hidden_dim, config.depth, bias=False)
            self.output_liear = nn.Linear(config.depth, config.hidden_dim, bias=False)
            self.dropout = nn.Dropout(0.0)
            self.fuse_weight1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.fuse_weight2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.fuse_weight1.data.fill_(1.0)
            self.fuse_weight2.data.fill_(0.0)
            """
            self.conv = PositionwiseFeedForward(
                config.hidden_dim,
                config.filter,
                config.hidden_dim,
                layer_config="cc",
                padding="both",
                dropout=0.0,
            )
            """
        else:
            self.emo_lin = nn.Linear(config.hidden_dim, decoder_number, bias=False)

        if config.emo_flat:
            self.emotion_enc = Emotion_Encoder(
                config.emb_dim,
                config.hidden_dim,
                num_layers=config.hop,
                num_heads=config.heads,
                total_key_depth=config.depth,
                total_value_depth=config.depth,
                filter_size=config.filter,
                universal=config.universal,
            )

        self.generator = Generator(config.hidden_dim, self.vocab_size)

        if config.dp_y_gen:
            self.dp_y_generator = Generator(config.hidden_dim, 50)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(
                size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1
            )
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(
                config.hidden_dim,
                1,
                8000,
                torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
            )

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=config.device)
            self.load_state_dict(state["model"])
            if load_optim:
                self.optimizer.load_state_dict(state["optimizer"])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter):
        state = {
            "iter": iter,
            "optimizer": self.optimizer.state_dict(),
            "current_loss": running_avg_ppl,
            "model": self.state_dict(),
        }
        model_save_path = os.path.join(
            self.model_dir,
            "BAKI_{}_{:.4f}".format(iter, running_avg_ppl),
        )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(
            shape[0], shape[1], config.heads, shape[2] // config.heads
        ).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (
            x.permute(0, 2, 1, 3)
            .contiguous()
            .view(shape[0], shape[2], shape[3] * config.heads)
        )

    def emo_enhance_tree(self, encoder_outputs_, rel_map, ctx_batch, emo_batch):
        # NOTE: 深拷贝, 另外开辟内存
        enhanced_encoder_outputs_ = encoder_outputs_.clone()
        # NOTE: 将CLS对应的隐藏层置零, 结果不甚理想
        #enhanced_encoder_outputs_[:,0] = torch.zeros(enhanced_encoder_outputs_.shape[0],enhanced_encoder_outputs_.shape[2])
        # [bz, len, emb] --> [bz, len, depth] --> [bz, n_heads, len, depth/n_heads]
        enhanced_queries = self._split_heads(self.query_linear(enhanced_encoder_outputs_))
        enhanced_keys = self._split_heads(self.key_linear(enhanced_encoder_outputs_))
        enhanced_values = self._split_heads(self.value_linear(enhanced_encoder_outputs_))
        query_scale = (config.depth // config.heads) ** -0.5
        enhanced_queries *= query_scale
        for i in range(enhanced_encoder_outputs_.shape[0]):
            dp_x = rel_map[i][0]
            op_orders = get_emo_enhance_op_order(dp_x, get_emotion_words_pos(ctx_batch[i], emo_batch[i]))
            # [bz, n, len, depth/n] --> [n, len, depth/n]
            enhanced_query = enhanced_queries[i]
            enhanced_key = enhanced_keys[i]
            enhanced_value = enhanced_values[i]
            for father, children in op_orders.items():
                # [n, len, depth/n]-->[n, depth/n]-->[n, 1, depth/n]
                hidden_q = enhanced_query[:,father].unsqueeze(1)
                # [n, len, depth/n]-->[n, child_num, depth/n], 把children对应的tensor选出来
                hidden_k = torch.index_select(enhanced_key, dim=1, index=torch.tensor(children).to(config.device))
                hidden_v = torch.index_select(enhanced_value, dim=1, index=torch.tensor(children).to(config.device))
                # [n, 1, depth/n] * [n, depth/n, child_num] = [n, 1, child_num]
                self_logits = torch.matmul(hidden_q.clone(), hidden_k.permute(0,2,1).clone())
                weights = nn.functional.softmax(self_logits, dim=-1)
                weights = self.dropout(weights)
                # [n, 1, child_num] * [n, child_num, depth/n] = [n, 1, depth/n] --> [n, depth/n]
                enhanced_hidden_f = torch.matmul(weights, hidden_v.clone()).squeeze(1)
                # 因为赋值只是换了个名字, 所以enhanced_query key value一起改变
                #print(enhanced_query[:,father].shape, enhanced_hidden_f.shape)
                enhanced_query[:,father] = enhanced_hidden_f
                enhanced_key[:, father] = enhanced_hidden_f
                enhanced_value[:, father] = enhanced_hidden_f
        # [bz, n, len, depth/n] --> [bz, len, emb]
        #print(enhanced_values.shape)
        enhanced_return = self.output_liear(self._merge_heads(enhanced_values))
        return enhanced_return

    def emo_enhance_flat(self, encoder_outputs_, ctx_batch, emo_batch):
        #print("[encoder_outputs_]:",encoder_outputs_.shape)
        #print("[ctx_batch]:",ctx_batch.shape)
        #print("[emo_batch]:",emo_batch.shape)
        enhanced_encoder_outputs_ = encoder_outputs_.clone()
        # [bz, len, emb] --> [bz, len, depth] --> [bz, n_heads, len, depth/n_heads]
        enhanced_queries = self._split_heads(self.query_linear(enhanced_encoder_outputs_))
        #enhanced_keys = self._split_heads(self.key_linear(enhanced_encoder_outputs_))
        #enhanced_values = self._split_heads(self.value_linear(enhanced_encoder_outputs_))
        query_scale = (config.depth // config.heads) ** -0.5
        enhanced_queries *= query_scale
        for i in range(enhanced_encoder_outputs_.shape[0]):
            cls_emb = enhanced_queries[i,:,0].unsqueeze(1)
            items = get_emotion_words_pos(ctx_batch[i], emo_batch[i])

            emotion_emb = torch.index_select(enhanced_queries[i],dim=1,index=torch.tensor(items).to(config.device))
            emotion_embs = torch.concat((cls_emb,emotion_emb),dim=1)
            hidden_q = emotion_embs
            hidden_k = emotion_embs
            hidden_v = emotion_embs
            self_logits = torch.matmul(hidden_q.clone(), hidden_k.permute(0, 2, 1).clone())
            weights = nn.functional.softmax(self_logits, dim=-1)
            weights = self.dropout(weights)
            enhanced_hidden_f = torch.matmul(weights, hidden_v.clone())
            enhanced_queries[i][:, 0] = enhanced_hidden_f[:, 0]
            for index1,index2 in enumerate(items):
                enhanced_queries[i][:, index2] = enhanced_hidden_f[:, index1]

        enhanced_return = self.output_liear(self._merge_heads(enhanced_queries))
        return enhanced_return

    def train_one_batch(self, batch, iter, train=True):

        enc_emo_batch = batch["emotion_context_batch"]

        (
            enc_batch,
            enc_rel_batch,
            enc_rel_map,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        dec_batch, _, _, _, _, _, _ = get_output_from_batch(batch)

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        mask_src_rel = enc_rel_batch.data.eq(config.UNK_DP).unsqueeze(2)

        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, self.embedding(enc_rel_batch), enc_rel_map, mask_src, mask_src_rel)
        if config.emo_enhance:
            if config.emo_tree:
                enhanced_encoder_outputs = self.emo_enhance_tree(encoder_outputs, enc_rel_map, enc_batch, enc_emo_batch)
                emo_logits = self.emo_lin(enhanced_encoder_outputs[:, 0])
            if config.emo_flat:
                enhanced_encoder_outputs = self.emo_enhance_flat(encoder_outputs, enc_batch, enc_emo_batch)
                emo_logits = self.emo_lin(enhanced_encoder_outputs[:, 0])
        else:
            emo_logits = self.emo_lin(encoder_outputs[:, 0])
        ## Emotion Identify
        emo_label = torch.LongTensor(batch["program_label"]).to(config.device)
        emo_loss = nn.CrossEntropyLoss()(emo_logits, emo_label)

        # Decode
        if config.prepend_emo:
            sos_emb = self.emotion_embedding(emo_logits).unsqueeze(1)
            dec_emb = self.embedding(dec_batch[:, :-1])
            dec_emb = torch.cat((sos_emb, dec_emb), dim=1)
            mask_trg = dec_batch.data.eq(config.PAD_idx).unsqueeze(1)
            pre_logit, attn_dist = self.decoder(dec_emb, self.fuse_weight1*encoder_outputs+self.fuse_weight2*enhanced_encoder_outputs, (mask_src, mask_trg))

        else:
            sos_token = (
                torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1)
            ).to(config.device)

            dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)

            mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
            """
            
            """
            if config.emo_enhance:
                pre_logit, attn_dist = self.decoder(
                    self.embedding(dec_batch_shift), self.fuse_weight1*encoder_outputs+self.fuse_weight2*enhanced_encoder_outputs, (mask_src, mask_trg)
                )
            else:
                pre_logit, attn_dist = self.decoder(
                    self.embedding(dec_batch_shift), encoder_outputs, (mask_src, mask_trg)
                )

        ## compute output dist
        logit = self.generator(
            pre_logit,
            attn_dist,
            enc_batch_extend_vocab if config.pointer_gen else None,
            extra_zeros,
            attn_dist_db=None,
        )

        ## ctx_loss: NNL if ptr else Cross entropy
        ctx_loss = self.criterion(
            logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1),
        )

        loss = ctx_loss + 1.0 * emo_loss

        loss_bce_program = nn.CrossEntropyLoss()(
            emo_logits, torch.LongTensor(batch["program_label"]).to(config.device)
        ).item()

        pred_program = np.argmax(emo_logits.detach().cpu().numpy(), axis=1)
        program_acc = accuracy_score(batch["program_label"], pred_program)

        if config.label_smoothing:
            loss_ppl = self.criterion_ppl(
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1),
            ).item()

        if train:
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            self.optimizer.step()

        if config.label_smoothing:
            return (
                loss_ppl,
                math.exp(min(loss_ppl, 100)),
                loss_bce_program,
                program_acc,
            )
        else:
            return (
                loss.item(),
                math.exp(min(loss.item(), 100)),
                loss_bce_program,
                program_acc,
            )

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        (
            enc_batch,
            enc_rel_batch,
            enc_rel_map,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        enc_emo_batch = batch["emotion_context_batch"]

        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        mask_src_rel = enc_rel_batch.data.eq(config.UNK_DP).unsqueeze(2)

        emb_mask = self.embedding(batch["mask_input"])
        # print("decoder_greedy:",enc_rel_map.shape)
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, self.embedding(enc_rel_batch), enc_rel_map, mask_src, mask_src_rel)
        if config.emo_enhance:
            if config.emo_tree:
                enhanced_encoder_outputs = self.emo_enhance_tree(encoder_outputs, enc_rel_map, enc_batch, enc_emo_batch)
                emo_logits = self.emo_lin(enhanced_encoder_outputs[:, 0])
            if config.emo_flat:
                enhanced_encoder_outputs = self.emo_enhance_flat(encoder_outputs, enc_batch, enc_emo_batch)
                emo_logits = self.emo_lin(enhanced_encoder_outputs[:, 0])
        else:
            ## Emotion Identify
            emo_logits = self.emo_lin(encoder_outputs[:, 0])

        if config.prepend_emo:
            ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
            ys_emb = self.emotion_embedding(emo_logits).unsqueeze(1)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
            decoded_words = []
            for i in range(max_dec_step + 1):
                if config.project:
                    out, attn_dist = self.decoder(
                        self.embedding_proj_in(self.embedding(ys)),
                        self.embedding_proj_in(encoder_outputs),
                        (mask_src, mask_trg),
                    )
                else:
                    out, attn_dist = self.decoder(ys_emb, self.fuse_weight1*encoder_outputs+self.fuse_weight2*enhanced_encoder_outputs, (mask_src, mask_trg))

                prob = self.generator(
                    out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
                )
                _, next_word = torch.max(prob[:, -1], dim=1)
                decoded_words.append(
                    [
                        "<EOS>"
                        if ni.item() == config.EOS_idx
                        else self.vocab.index2word[ni.item()]
                        for ni in next_word.view(-1)
                    ]
                )
                next_word = next_word.data[0]

                ys = torch.cat(
                    [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                    dim=1,
                ).to(config.device)
                ys_emb = torch.cat(
                    (
                        ys_emb,
                        self.embedding(torch.ones(1, 1).long().fill_(next_word).to(config.device))
                    ),
                    dim=1,
                )
                mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        else:
            ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

            decoded_words = []

            for i in range(max_dec_step + 1):
                word_list = []

                for w in decoded_words:
                    word_list += w

                if config.project:
                    print("project=True")
                    out, attn_dist = self.decoder(
                        self.embedding_proj_in(self.embedding(ys)),
                        self.embedding_proj_in(enhanced_encoder_outputs),
                        (mask_src, mask_trg),
                    )
                else:
                    if config.emo_enhance:
                        out, attn_dist = self.decoder(
                            self.embedding(ys), self.fuse_weight1*encoder_outputs+self.fuse_weight2*enhanced_encoder_outputs, (mask_src, mask_trg)
                        )
                    else:
                        out, attn_dist = self.decoder(
                            self.embedding(ys),
                            encoder_outputs,
                            (mask_src, mask_trg)
                        )

                prob = self.generator(
                    out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
                )

                _, next_word = torch.max(prob[:, -1], dim=1)
                decoded_words.append(
                    [
                        "<EOS>"
                        if ni.item() == config.EOS_idx
                        else self.vocab.index2word[ni.item()]
                        for ni in next_word.view(-1)
                    ]
                )
                next_word = next_word.data[0]

                ys = torch.cat(
                    [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                    dim=1,
                ).to(config.device)
                mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent

    def decoder_topk(self, batch, max_dec_step=30):
        (
            enc_batch,
            enc_rel_batch,
            enc_rel_map,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, self.embedding(enc_rel_batch), enc_rel_map,
                                       mask_src)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(encoder_outputs),
                    (mask_src, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    self.embedding(ys), encoder_outputs, (mask_src, mask_trg)
                )

            logit = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            filtered_logit = top_k_top_p_filtering(
                # in CEM: logit[0, -1] / 0.7, top_k=0, top_p=0.9, filter_value=-float("Inf")
                logit[:, -1], top_k=3, top_p=0, filter_value=-float("Inf")
            )
            # Sample from the filtered distribution
            next_word = torch.multinomial(
                F.softmax(filtered_logit, dim=-1), 1
            ).squeeze()
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            next_word = next_word.data[0]

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent

