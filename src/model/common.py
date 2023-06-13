import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter
from src.utils import config
#from src.model.common import PositionwiseFeedForward, LayerNorm
from tqdm import tqdm
if config.model == "EmoDT":
    from src.utils.decode.emodt import Translator


class DeprelAttention(nn.Module):
    """
    Reference: <Integrating Dependency Tree Into Self-attention for Sentence Representation>
    """
    def __init__(
        self,
        input_depth, ##config.hidden_dim
        total_key_depth, #config.depth
        total_value_depth, #config.depth
        # NOTE:
        total_rel_depth, #config.depth
        emb_depth, ##config.hidden_dim
        rel_depth, ##config.emb_dim
        output_depth, #config.hidden_dim
        num_heads, #
        bias_mask=None,
        dropout=0.0,
    ):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(DeprelAttention, self).__init__()

        if total_key_depth % num_heads != 0:
            print(
                "Key depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_key_depth, num_heads)
            )
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print(
                "Value depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_value_depth, num_heads)
            )
            total_value_depth = total_value_depth - (total_value_depth % num_heads)

        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5  ## sqrt
        self.bias_mask = bias_mask

        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)
        # relation_linear转换关系词向量的维度, 并通过分头操作, 将其映射到子空间; vector_linear提供一个可学习向量, 计算关系注意力分数
        #self.relation_linear = nn.Linear(input_depth, total_rel_depth, bias=False)
        #self.vector_linear = nn.Linear(1, total_rel_depth, bias=False)
        self.Vr = Parameter(torch.Tensor(rel_depth,1))

        # NOTE:
        self.Wge = nn.Linear(emb_depth, rel_depth, bias=False) #[config.hidden_dim, config.emb_dim]
        self.Wgr = nn.Linear(rel_depth, rel_depth, bias=False) #[config.emb_dim, config.emb_dim]
        #self.Wge = Parameter(torch.Tensor(emb_depth, rel_depth))
        #self.Wgr = Parameter(torch.Tensor(rel_depth, rel_depth))
        self.Vg = Parameter(torch.Tensor(rel_depth,1)) #[config.emb_dim, 1]
        self.act = nn.Tanh()

        self.dropout = nn.Dropout(dropout)

    # NOTE: [bz, len, depth]
    def _semantic_gated_mechanism(self, hidden_states, relation_vectors, dp_map):
        """hidden_states is encoder_outputs: [bz, len, hidden_dim]"""
        #print("##############now in _semantic_gated_mechanism!")
        emb_depth = hidden_states.shape[2]
        rel_depth = relation_vectors.shape[2]
        # Do a linear for each component
        # print("hidden_states.shape:",hidden_states.shape)
        # hidden_states = self.Wge(hidden_states)
        # relation_vectors = self.Wgr(relation_vectors)
        # Split into multiple heads
        #print("relation_vectors.shape:", relation_vectors.shape)
        #relation_vectors = torch.matmul(relation_vectors, self.Wgr)
        relation_vectors = self.Wgr(relation_vectors)
        relation_vectors = self._split_heads(relation_vectors)
        #hidden_states = torch.matmul(hidden_states, self.Wge)
        hidden_states = self.Wge(hidden_states)
        hidden_states = self._split_heads(hidden_states)
        Vg = self.Vg.view(self.num_heads, rel_depth // self.num_heads, 1)
        # NOTE: [bz, n_head, len, rel_depth/n_head] * [n_heads, rel_depth/n_head, 1] = [bz, n_heads, len, 1]
        #print("[hidden_states]:",hidden_states.shape)
        #print("[relation_vectors]:",relation_vectors.shape)
        gated_scores = self.act(torch.matmul(hidden_states + relation_vectors, Vg))

        # NOTE: [bz, n_heads, len, 1] --> [bz, n_heads, len, len]
        #print("gated_scores.shape1:", gated_scores.shape)
        gated_scores = self._sparse(gated_scores, dp_map)
        #print("gated_scores.shape2:", gated_scores.shape)
        return gated_scores

    # NOTE: att: [bz, n_heads, len, 1] --> new_att: [bz, n_heads, len, len]
    def _sparse(self, att_tensor, pos_tensor):
        #print("##############now in _sparse!")
        att_tensor = att_tensor.squeeze(-1)
        # print(att_tensor.shape)
        bz, n_heads, length = att_tensor.shape
        #print("att_tensor:", att_tensor.shape)
        #print("pos_tensor:", pos_tensor.shape)

        bz_index = torch.arange(start=0, end=bz, step=1).repeat_interleave(n_heads * length, dim=0).to(config.device)
        head_index = torch.arange(start=0, end=n_heads, step=1).repeat_interleave(length, dim=0).repeat(bz).to(config.device)
        # NOTE: [bz, 2, len] --> [2, bz*n_heads*len] then add to index[2] index[3]
        #print(pos_tensor)
        #print(pos_tensor.repeat_interleave(n_heads, dim=0).chunk(2, dim=1))
        #print(length)
        pos_index = torch.cat(pos_tensor.repeat_interleave(n_heads, dim=0).chunk(2, dim=1), dim=0).view(2,
                                                                                                        bz * n_heads * length).to(config.device)
        #print("[pos_index]:",pos_index.shape)
        # NOTE: 注意力权重对应的坐标
        index = (bz_index, head_index, pos_index[0], pos_index[1])
        # NOTE: initialize
        sparse_tensor = torch.zeros([bz, n_heads, length, length], dtype=torch.float).to(config.device)
        # NOTE: 将注意力权值填入index指定的位置
        sparse_tensor.index_put_(index, att_tensor.view(-1))
        # NOTE: 用sparse_tensor的转置代表对称位置上的关系注意力
        sparse_tensor_t = torch.transpose(input=sparse_tensor, dim0=3, dim1=2)
        sparse_tensor = sparse_tensor + sparse_tensor_t
        #print("YES!")
        return sparse_tensor

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
            shape[0], shape[1], self.num_heads, shape[2] // self.num_heads
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
            .view(shape[0], shape[2], shape[3] * self.num_heads)
        )

    def reverse_(self, hidden_states, dp_map):
        """using reversed hidden_states in _semantic_gated_mechanism"""
        def rev_one(one_state, map_x):
            # NOTE: [len, depth], [len, ]
            r = torch.zeros_like(one_state)
            # NOTE: 跳过起始的CLS
            for j in range(1,one_state.shape[0]):
                r[j] = one_state[map_x[j]]
            return r
        # NOTE: hidden_states = [bz, len, depth], dp_map = [bz, 2, len]
        value_r = torch.zeros_like(hidden_states)
        for i in range(hidden_states.shape[0]):
            value_r[i] = rev_one(hidden_states[i], dp_map[i][0])

        return value_r

    def forward(self, queries, keys, values, relations, dp_map, mask, rel_mask):
        #print("[values]：",values.shape)
        #print("[relations]：",relations.shape)
        # NOTE: 计算gated_score时需要将hidden_states与relations做掩码处理
        # NOTE: 因为gated_score是关于relation的得分, 所以hidden_states应用relmask而非mask
        if config.is_reverse:
            #print("is_reverse!")
            hidden_values = self.reverse_(values, dp_map).masked_fill(rel_mask, 0)
        else:
            hidden_values = values.masked_fill(rel_mask, 0)
        # NOTE: [bz, len, emb].masked_fill([bz, len, 1])
        relations = relations.masked_fill(rel_mask, 0)
        gated_scores = self._semantic_gated_mechanism(hidden_values, relations, dp_map)
        # Do a linear for each component
        # [bz, len, depth]
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)


        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)
        relations = self._split_heads(relations)

        learnable_vector = self.Vr.view(self.num_heads, self.Vr.shape[0] // self.num_heads, 1)

        # Scale queries
        queries *= self.query_scale

        # Combine queries and keys
        # NOTE: Self Attention Scoring
        self_logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        #print("relations.shape:",relations.shape)
        #print("learnable_vector.shape:",learnable_vector.permute(0, 2, 1).shape)
        # NOTE: Relation Scoring
        relation_logits = torch.matmul(relations, learnable_vector)
        relation_logits = self._sparse(relation_logits, dp_map)

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, len]
            self_logits = self_logits.masked_fill(mask, -1e18)
            relation_logits = relation_logits.masked_fill(mask, -1e18)

        ## attention weights
        attetion_weights = self_logits.sum(dim=1) / self.num_heads

        # Convert to probabilites
        self_weights = nn.functional.softmax(self_logits, dim=-1)
        # NOTE:
        relation_weights = nn.functional.softmax(relation_logits, dim=-1)
        #gated_scores = self._semantic_gated_mechanism(values, relations, dp_map)
        # [bz, n_heads, len, len].[bz, n_heads, len, len] + [bz, n_heads, len, len].[bz, n_heads, len, len] = [bz, n_heads, len, len]
        # NOTE: torch.mul 是按位相乘, self_weights和relation_weights已做过掩码处理, gated_scores不必再做
        weights = torch.mul(1-gated_scores, self_weights) + torch.mul(gated_scores, relation_weights)

        # Dropout
        weights = self.dropout(weights)

        # Combine with values to get context
        contexts = torch.matmul(weights, values)

        # Merge heads
        contexts = self._merge_heads(contexts)

        # Linear to get output
        outputs = self.output_linear(contexts)

        return outputs, attetion_weights

class CommonEncoderLayer(nn.Module):
    """
    Represents one Encoder layer of the Transformer Encoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(
        self,
        hidden_size,
        total_key_depth,
        total_value_depth,
        filter_size,
        num_heads,
        bias_mask=None,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(CommonEncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            bias_mask,
            attention_dropout,
        )

        self.positionwise_feed_forward = PositionwiseFeedForward(
            hidden_size,
            filter_size,
            hidden_size,
            layer_config="cc",
            padding="both",
            dropout=relu_dropout,
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs, mask=None):
        x = inputs

        # Layer Normalization
        x_norm = self.layer_norm_mha(x)

        # Multi-head attention
        y, _ = self.multi_head_attention(x_norm, x_norm, x_norm, mask)

        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual
        y = self.dropout(x + y)

        return y

class EncoderLayer(nn.Module):
    """
    Represents one Encoder layer of the Transformer Encoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(
        self,
        embedding_size, #config.emb_dim
        hidden_size, #config.hidden_dim
        total_key_depth, #config.depth
        total_value_depth, #config.depth
        total_rel_depth, #config.depth
        filter_size,
        num_heads, #config.heads
        bias_mask=None,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(EncoderLayer, self).__init__()

        self.dependency_relation_attention = DeprelAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            total_rel_depth,
            hidden_size,# emb_depth
            embedding_size,# rel_depth
            hidden_size,# output_depth
            num_heads,
            bias_mask,
            attention_dropout,
        )

        self.positionwise_feed_forward = PositionwiseFeedForward(
            hidden_size,
            filter_size,
            hidden_size,
            layer_config="cc",
            padding="both",
            dropout=relu_dropout,
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs, relations, dp_map, mask=None, rel_mask=None):
        #print("now in encoder_layer!")
        x = inputs

        # Layer Normalization
        x_norm = self.layer_norm_mha(x)

        # Multi-head attention
        y, _ = self.dependency_relation_attention(x_norm, x_norm, x_norm, relations, dp_map, mask, rel_mask)

        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual
        y = self.dropout(x + y)

        return y


class DecoderLayer(nn.Module):
    """
    Represents one Decoder layer of the Transformer Decoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(
        self,
        hidden_size,
        total_key_depth,
        total_value_depth,
        filter_size,
        num_heads,
        bias_mask,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(DecoderLayer, self).__init__()

        self.multi_head_attention_dec = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            bias_mask, # need to hide the future
            attention_dropout,
        )

        self.multi_head_attention_enc_dec = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            None, # no need to hide the future
            attention_dropout,
        )

        self.positionwise_feed_forward = PositionwiseFeedForward(
            hidden_size,
            filter_size,
            hidden_size,
            layer_config="cc",
            padding="left",
            dropout=relu_dropout,
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        """
        NOTE: Inputs is a tuple consisting of decoder inputs and encoder output
        """

        x, encoder_outputs, attention_weight, mask = inputs
        # NOTE: mask_src only mask PADs in src. dec_mask mask both PADs and subsequence in tgt.
        mask_src, dec_mask = mask

        # Layer Normalization before decoder self attention
        x_norm = self.layer_norm_mha_dec(x)

        # Masked Multi-head attention
        # NOTE: self_attention use dec_mask(to mask both PADs and subsequence in tgt)
        y, _ = self.multi_head_attention_dec(x_norm, x_norm, x_norm, dec_mask)

        # Dropout and residual after self-attention
        x = self.dropout(x + y)

        # Layer Normalization before encoder-decoder attention
        x_norm = self.layer_norm_mha_enc(x)

        # Multi-head encoder-decoder attention
        # NOTE: enc_dec_attention use mask_src(to mmask PADs in encoder_outputs)
        y, attention_weight = self.multi_head_attention_enc_dec(
            x_norm, encoder_outputs, encoder_outputs, mask_src
        )

        # Dropout and residual after encoder-decoder attention
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual after positionwise feed forward layer
        y = self.dropout(x + y)

        # Return encoder outputs as well to work with nn.Sequential
        return y, encoder_outputs, attention_weight, mask


class MultiExpertMultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_experts,
        input_depth,
        total_key_depth,
        total_value_depth,
        output_depth,
        num_heads,
        bias_mask=None,
        dropout=0.0,
    ):
        """
        Parameters:
            expert_num: Number of experts
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiExpertMultiHeadAttention, self).__init__()
        # Checks borrowed from
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

        if total_key_depth % num_heads != 0:
            print(
                "Key depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_key_depth, num_heads)
            )
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print(
                "Value depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_value_depth, num_heads)
            )
            total_value_depth = total_value_depth - (total_value_depth % num_heads)
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5  ## sqrt
        self.bias_mask = bias_mask

        # Key and query depth will be same
        self.query_linear = nn.Linear(
            input_depth, total_key_depth * num_experts, bias=False
        )
        self.key_linear = nn.Linear(
            input_depth, total_key_depth * num_experts, bias=False
        )
        self.value_linear = nn.Linear(
            input_depth, total_value_depth * num_experts, bias=False
        )
        self.output_linear = nn.Linear(
            total_value_depth, output_depth * num_experts, bias=False
        )

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_experts ,num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(
            shape[0],
            shape[1],
            self.num_experts,
            self.num_heads,
            shape[2] // (self.num_heads * self.num_experts),
        ).permute(0, 2, 3, 1, 4)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_experts ,num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 5:
            raise ValueError("x must have rank 5")
        shape = x.shape
        return (
            x.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(shape[0], shape[3], self.num_experts, shape[4] * self.num_heads)
        )

    def forward(self, queries, keys, values, mask):

        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # Scale queries
        queries *= self.query_scale

        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 2, 4, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 1, T_values]
            logits = logits.masked_fill(mask, -1e18)

        ## attention weights
        # attetion_weights = logits.sum(dim=1)/self.num_heads

        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)

        # Dropout
        weights = self.dropout(weights)

        # Combine with values to get context
        contexts = torch.matmul(weights, values)

        # Merge heads
        contexts = self._merge_heads(contexts)
        # contexts = torch.tanh(contexts)

        # Linear to get output
        outputs = self.output_linear(contexts)

        return outputs


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """

    def __init__(
        self,
        input_depth,
        total_key_depth,
        total_value_depth,
        output_depth,
        num_heads,
        bias_mask=None,
        dropout=0.0,
    ):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        # Checks borrowed from
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

        if total_key_depth % num_heads != 0:
            print(
                "Key depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_key_depth, num_heads)
            )
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print(
                "Value depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_value_depth, num_heads)
            )
            total_value_depth = total_value_depth - (total_value_depth % num_heads)

        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5  ## sqrt
        self.bias_mask = bias_mask

        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)

        self.dropout = nn.Dropout(dropout)

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
            shape[0], shape[1], self.num_heads, shape[2] // self.num_heads
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
            .view(shape[0], shape[2], shape[3] * self.num_heads)
        )

    def forward(self, queries, keys, values, mask):

        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # Scale queries
        queries *= self.query_scale

        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            logits = logits.masked_fill(mask, -1e18)

        ## attention weights
        attetion_weights = logits.sum(dim=1) / self.num_heads

        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)

        # Dropout
        weights = self.dropout(weights)

        # Combine with values to get context
        contexts = torch.matmul(weights, values)

        # Merge heads
        contexts = self._merge_heads(contexts)

        # Linear to get output
        outputs = self.output_linear(contexts)

        return outputs, attetion_weights


class Conv(nn.Module):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """

    def __init__(self, input_size, output_size, kernel_size, pad_type):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data),
                      both -> pad on both sides
        """
        super(Conv, self).__init__()
        padding = (
            (kernel_size - 1, 0)
            if pad_type == "left"
            else (kernel_size // 2, (kernel_size - 1) // 2)
        )
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(
            input_size, output_size, kernel_size=kernel_size, padding=0
        )

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)

        return outputs


class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """

    def __init__(
        self,
        input_depth,
        filter_size,
        output_depth,
        layer_config="ll",
        padding="left",
        dropout=0.0,
    ):
        """
        Parameters:
            input_depth: Size of last dimension of input
            filter_size: Hidden size of the middle layer
            output_depth: Size last dimension of the final output
            layer_config: ll -> linear + ReLU + linear
                          cc -> conv + ReLU + conv etc.
            padding: left -> pad on the left side (to mask future data),
                     both -> pad on both sides
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(PositionwiseFeedForward, self).__init__()

        layers = []
        sizes = (
            [(input_depth, filter_size)]
            + [(filter_size, filter_size)] * (len(layer_config) - 2)
            + [(filter_size, output_depth)]
        )

        for lc, s in zip(list(layer_config), sizes):
            if lc == "l":
                layers.append(nn.Linear(*s))
            elif lc == "c":
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    # Borrowed from jekbradbury
    # https://github.com/pytorch/pytorch/issues/1959
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)

    return torch_mask.unsqueeze(0).unsqueeze(1)


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    生成位置编码
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        float(num_timescales) - 1
    )
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype(np.float) * -log_timescale_increment
    )
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(
        signal, [[0, 0], [0, channels % 2]], "constant", constant_values=[0.0, 0.0]
    )
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


def _get_attn_subsequent_mask(size):
    """
    Get an attention mask to avoid using the subsequent info.
    Args:
        size: int
    Returns:
        (`LongTensor`):
        * subsequent_mask `[1 x size x size]`
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    subsequent_mask = torch.from_numpy(subsequent_mask)

    return subsequent_mask.to(config.device)


class OutputLayer(nn.Module):
    """
    Abstract base class for output layer.
    Handles projection to output labels
    """

    def __init__(self, hidden_size, output_size):
        super(OutputLayer, self).__init__()
        self.output_size = output_size
        self.output_projection = nn.Linear(hidden_size, output_size)

    def loss(self, hidden, labels):
        raise NotImplementedError(
            "Must implement {}.loss".format(self.__class__.__name__)
        )


class SoftmaxOutputLayer(OutputLayer):
    """
    Implements a softmax based output layer
    """

    def forward(self, hidden):
        logits = self.output_projection(hidden)
        probs = F.softmax(logits, -1)
        _, predictions = torch.max(probs, dim=-1)

        return predictions

    def loss(self, hidden, labels):
        logits = self.output_projection(hidden)
        log_probs = F.log_softmax(logits, -1)
        return F.nll_loss(log_probs.view(-1, self.output_size), labels.view(-1))


def position_encoding(sentence_size, embedding_dim):
    encoding = np.ones((embedding_dim, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_dim + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (embedding_dim + 1) / 2) * (
                j - (sentence_size + 1) / 2
            )
    encoding = 1 + 4 * encoding / embedding_dim / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    # encoding[:, -1] = 1.0
    return np.transpose(encoding)


def gen_embeddings(vocab):
    """
    Generate an initial embedding matrix for `word_dict`.
    If an embedding file is not given or a word is not in the embedding file,
    a randomly initialized vector will be used.
    """
    embeddings = np.random.randn(vocab.n_words, config.emb_dim) * 0.01
    print("Embeddings: %d x %d" % (vocab.n_words, config.emb_dim))
    if config.emb_file is not None:
        print("Loading embedding file: %s" % config.emb_file)
        pre_trained = 0
        for line in open(config.emb_file, encoding='utf-8').readlines():
            sp = line.split()
            if len(sp) == config.emb_dim + 1:
                if sp[0] in vocab.word2index:
                    pre_trained += 1
                    embeddings[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print(sp[0])
        print(
            "Pre-trained: %d (%.2f%%)"
            % (pre_trained, pre_trained * 100.0 / vocab.n_words)
        )
    return embeddings


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model, padding_idx=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


def share_embedding(vocab, pretrain=True):
    embedding = Embeddings(vocab.n_words, config.emb_dim, padding_idx=config.PAD_idx)
    if pretrain:
        pre_embedding = gen_embeddings(vocab)
        embedding.lut.weight.data.copy_(torch.FloatTensor(pre_embedding))
        embedding.lut.weight.data.requires_grad = True
    return embedding


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        # NOTE: 初始化与x形状相同, 用smooth填充
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        # NOTE: .scatter_()得到target的one-hot编码: true_dist=[bz*len, vocab_size]
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.size()[0] > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        return self.optimizer.state_dict()

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(config.PAD_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_input_from_batch(batch):
    enc_batch = batch["input_batch"]
    enc_rel_batch = batch["input_rel_batch"]
    #print("[input_batch]:",enc_batch,"[input_txt]:",batch["input_txt"],"[input_rel_batch]:",enc_rel_batch)
    enc_rel_map = batch["input_dp_map"]
    enc_lens = batch["input_lengths"]
    enc_txt = batch["input_txt"]
    batch_size, max_enc_len = enc_batch.size()
    assert len(enc_lens) == batch_size

    enc_padding_mask = sequence_mask(enc_lens, max_len=max_enc_len).float()

    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        print("config.pointer_gen!")
        enc_batch_extend_vocab = batch["input_ext_vocab_batch"]
        # max_art_oovs is the max over all the article oov list in the batch
        if batch["max_art_oovs"] > 0:
            extra_zeros = torch.zeros((batch_size, batch["max_art_oovs"]))

    c_t_1 = torch.zeros((batch_size, 2 * config.hidden_dim))

    coverage = None
    if config.is_coverage:
        coverage = torch.zeros(enc_batch.size()).to(config.device)

    enc_padding_mask.to(config.device)
    if enc_batch_extend_vocab is not None:
        enc_batch_extend_vocab.to(config.device)
    if extra_zeros is not None:
        extra_zeros.to(config.device)
    c_t_1.to(config.device)

    return (
        enc_batch,
        enc_rel_batch,
        enc_rel_map,
        enc_padding_mask,
        enc_lens,
        enc_batch_extend_vocab,
        extra_zeros,
        c_t_1,
        coverage,
        enc_txt,
    )


def get_output_from_batch(batch):

    dec_batch = batch["target_batch"]
    dec_rel_batch = batch["target_rel_batch"]
    dec_rel_map = batch["target_dp_map"]

    if config.pointer_gen:
        target_batch = batch["target_ext_vocab_batch"]
    else:
        target_batch = dec_batch

    dec_lens_var = batch["target_lengths"]
    max_dec_len = max(dec_lens_var)

    assert max_dec_len == target_batch.size(1)

    dec_padding_mask = sequence_mask(dec_lens_var, max_len=max_dec_len).float()

    return dec_batch, dec_rel_batch, dec_rel_map, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

def get_output_from_batch_v2(batch):

    dec_batch = batch["target_batch"]
    dec_rel_batch = batch["target_rel_batch"]
    dec_rel_f_batch = batch["target_rel_f_batch"] # relation of current word's father

    dec_rel_map = batch["target_dp_map"]

    if config.pointer_gen:
        target_batch = batch["target_ext_vocab_batch"]
    else:
        target_batch = dec_batch

    dec_lens_var = batch["target_lengths"]
    max_dec_len = max(dec_lens_var)

    assert max_dec_len == target_batch.size(1)

    dec_padding_mask = sequence_mask(dec_lens_var, max_len=max_dec_len).float()

    return dec_batch, dec_rel_batch, dec_rel_f_batch, dec_rel_map, dec_padding_mask, max_dec_len, dec_lens_var, target_batch


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = seq_range_expand
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.to(config.device)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


def write_config():
    if not config.test:
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        with open(config.save_path + "config.txt", "w") as the_file:
            for k, v in config.args.__dict__.items():
                if "False" in str(v):
                    pass
                elif "True" in str(v):
                    the_file.write("--{} ".format(k))
                else:
                    the_file.write("--{} {} ".format(k, v))


def print_custum(emotion, dial, ref, hyp_b, hyp_g, pred_emotions, comet_res):
    res = ""
    res += "Emotion: {}".format(emotion) + "\n"
    if pred_emotions:
        res += "Pred Emotions: {}".format(pred_emotions) + "\n"
    if comet_res:
        for k, v in comet_res.items():
            res += "{}:{}".format(k, v) + "\n"
    res += "Context:{}".format(dial) + "\n"
    if hyp_b:
        res += "Beam:{}".format(hyp_b) + "\n"
    res += "Greedy:{}".format(hyp_g) + "\n"
    res += "Ref:{}".format(ref) + "\n"
    res += "---------------------------------------------------------------" + "\n"

    return res


def evaluate(model, data, ty="valid", max_dec_step=30):
    model.__id__logger = 0
    ref, hyp_g, results = [], [], []
    if ty == "test":
        print("testing generation:")
    l = []
    p = []
    bce = []
    acc = []
    top_preds = []
    comet_res = []
    pbar = tqdm(enumerate(data), total=len(data))

    if config.model != "v2" and config.model != "v3":
        t = Translator(model, model.vocab)
    for j, batch in pbar:
        # NOTE: 这里的train_one_batch梯度不回传
        loss, ppl, bce_prog, acc_prog = model.train_one_batch(
            batch, 0, train=False
        )
        l.append(loss)
        p.append(ppl)
        bce.append(bce_prog)
        acc.append(acc_prog)
        if ty == "test":
            sent_g = model.decoder_greedy(batch, max_dec_step=max_dec_step)
            if config.model != "v2" and config.model != "v3":
                sent_b = t.beam_search(batch, max_dec_step=max_dec_step)
            for i, greedy_sent in enumerate(sent_g):
                rf = " ".join(batch["target_txt"][i])
                hyp_g.append(greedy_sent)
                ref.append(rf)
                temp = print_custum(
                    emotion=batch["program_txt"][i],
                    dial=[" ".join(s) for s in batch["input_txt"][i]],
                    ref=rf,
                    hyp_b=sent_b[i] if config.model != "v2" and config.model != "v3" else "",
                    hyp_g=greedy_sent,
                    pred_emotions=top_preds,
                    comet_res=comet_res,
                )
                results.append(temp)
        pbar.set_description(
            "loss:{:.4f} ppl:{:.1f}".format(np.mean(l), math.exp(np.mean(l)))
        )

    loss = np.mean(l)
    ppl = np.mean(p)
    bce = np.mean(bce)
    acc = np.mean(acc)

    print("EVAL\tLoss\tPPL\tAccuracy\n")
    print("{}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ty, loss, math.exp(loss), acc))

    return loss, math.exp(loss), bce, acc, results


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits