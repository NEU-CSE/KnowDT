import os
import nltk
import json
import torch
import pickle
import logging
import numpy as np
from tqdm.auto import tqdm
from src.utils import config
import torch.utils.data as data
from src.utils.common import save_config
from nltk.corpus import wordnet, stopwords
from src.utils.constants import DATA_FILES
from src.utils.constants import EMO_MAP as emo_map
from src.utils.constants import WORD_PAIRS as word_pairs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


emotion_lexicon = json.load(open("data/NRCDict.json"))[0]
stop_words = stopwords.words("english")


class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def process_sent(sentence):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence


def encode_ctx(vocab, items, data_dict):
    for ctx in tqdm(items):
        ctx_list = []
        e_list = []
        for i, c in enumerate(ctx):
            item = process_sent(c)
            ctx_list.append(item)
            vocab.index_words(item) # add words in item to vocab.
            ws_pos = nltk.pos_tag(item)
            for w in ws_pos:
                w_p = get_wordnet_pos(w[1])
                if w[0] not in stop_words and (
                    w_p == wordnet.ADJ or w[0] in emotion_lexicon
                ):
                    e_list.append(w[0])

        data_dict["context"].append(ctx_list)
        data_dict["emotion_context"].append(e_list)


def encode(vocab, files):
    data_dict = {
        "context": [],
        "target": [],
        "emotion": [],
        "situation": [],
        "emotion_context": [],
        "utt_cs": [],
    }
    for i, k in enumerate(data_dict.keys()):
        items = files[i]
        if k == "context":
            encode_ctx(vocab, items, data_dict)
        elif k == "emotion":
            data_dict[k] = items
        else:
            for item in tqdm(items):
                item = process_sent(item)
                data_dict[k].append(item)
                vocab.index_words(item) # add words in item to vocab.
        if i == 3:
            break
    assert (
        len(data_dict["context"])
        == len(data_dict["target"])
        == len(data_dict["emotion"])
        == len(data_dict["situation"])
        == len(data_dict["emotion_context"])
        == len(data_dict["utt_cs"])
    )

    return data_dict


def read_files(vocab):
    files = DATA_FILES(config.data_dir)
    train_files = [np.load(f, allow_pickle=True) for f in files["train"]]
    dev_files = [np.load(f, allow_pickle=True) for f in files["dev"]]
    test_files = [np.load(f, allow_pickle=True) for f in files["test"]]
    data_train = encode(vocab, train_files)
    data_dev = encode(vocab, dev_files)
    data_test = encode(vocab, test_files)

    return data_train, data_dev, data_test, vocab


def load_dataset():
    data_dir = config.data_dir
    cache_file = f"{data_dir}/dataset_preproc_dp.p"
    if os.path.exists(cache_file):
        print("LOADING empathetic_dialogue")
        with open(cache_file, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    return data_tra, data_val, data_tst, vocab


class Dataset(data.Dataset):
    def __init__(self, data, vocab):
        self.vocab = vocab
        self.data = data
        self.emo_map = emo_map
        self.analyzer = SentimentIntensityAnalyzer()

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        item = {}
        item["context_text"] = self.data["context"][index]
        item["situation_text"] = self.data["situation"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]
        item["emotion_context"] = self.data["emotion_context"][index]

        item["context_dp_text"] = self.data["context_dp_name"][index]
        item["context_dp_x"] = self.data["context_dp_x"][index]
        item["context_dp_y"] = self.data["context_dp_y"][index]
        item["context_dp_map"] = self.data["context_dp_map"][index]
        item["target_dp_text"] = self.data["target_dp_name"][index]
        item["target_dp_f_text"] = self.data["target_dp_f_name"][index]
        item["target_dp_x"] = self.data["target_dp_x"][index]
        item["target_dp_y"] = self.data["target_dp_y"][index]
        item["target_dp_map"] = self.data["target_dp_map"][index]

        item["context_emotion_scores"] = self.analyzer.polarity_scores(
            " ".join(self.data["context"][index][0])
        )

        item["context"], item["context_mask"] = self.preprocess(item["context_text"])
        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["emotion"], item["emotion_label"] = self.preprocess_emo(
            item["emotion_text"], self.emo_map
        )
        (
            item["emotion_context"],
            item["emotion_context_mask"],
        ) = self.preprocess(item["emotion_context"], emo=True)

        item["context_dp"] = self.preprocess(item["context_dp_text"], ctx_dp=True)
        item["target_dp"] = self.preprocess(item["target_dp_text"], tgt_dp=True)
        item["target_dp_f"] = self.preprocess(item["target_dp_f_text"], tgt_dp=True)

        return item


    def preprocess(self, arr, anw=False, cs=None, emo=False, ctx_dp=False, tgt_dp=False):
        if anw:
            sequence = [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else config.UNK_idx
                for word in arr
            ] + [config.EOS_idx]

            return torch.LongTensor(sequence)
        elif emo:
            x_emo = [config.CLS_idx]
            x_emo_mask = [config.CLS_idx]
            for i, ew in enumerate(arr):
                x_emo += [
                    self.vocab.word2index[ew]
                    if ew in self.vocab.word2index
                    else config.UNK_idx
                ]
                x_emo_mask += [self.vocab.word2index["CLS"]]

            assert len(x_emo) == len(x_emo_mask)
            return torch.LongTensor(x_emo), torch.LongTensor(x_emo_mask)


        elif tgt_dp:
            dp_IDs = [
                self.vocab.word2index[dp_name]
                if dp_name in self.vocab.word2index
                else config.UNK_DP
                for dp_name in arr
            ] + [config.UNK_DP]
            return torch.LongTensor(dp_IDs)

        elif ctx_dp:
            dp_IDs = [config.UNK_DP]
            dp_IDs += [
                self.vocab.word2index[dp_name]
                if dp_name in self.vocab.word2index
                else config.UNK_DP
                for dp_name in arr
            ]
            return torch.LongTensor(dp_IDs)

        else:
            x_dial = [config.CLS_idx]
            x_mask = [config.CLS_idx]
            for i, sentence in enumerate(arr):
                x_dial += [
                    self.vocab.word2index[word]
                    if word in self.vocab.word2index
                    else config.UNK_idx
                    for word in sentence
                ]
                spk = (
                    self.vocab.word2index["USR"]
                    if i % 2 == 0
                    else self.vocab.word2index["SYS"]
                )
                x_mask += [spk for _ in range(len(sentence))]
            assert len(x_dial) == len(x_mask)

            return torch.LongTensor(x_dial), torch.LongTensor(x_mask)

    def preprocess_emo(self, emotion, emo_map):
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]


def collate_fn(data):
    def dp_map_merge(dp_maps):
        lengths = [len(dp_map[0]) for dp_map in dp_maps]
        max_l = max(lengths) + 1
        bz = len(dp_maps)
        padded_dp_x = torch.tensor([0]*max_l*bz).view(bz,1,max_l).long()
        padded_dp_y = torch.tensor([0]*max_l*bz).view(bz,1,max_l).long()
        for i, dp_x_y in enumerate(dp_maps):
            end = lengths[i]
            dp_x = torch.tensor(dp_x_y[0])
            dp_y = torch.tensor(dp_x_y[1])

            padded_dp_x[i, 0, 1:end+1] = dp_x[:end]
            padded_dp_y[i, 0, 1:end+1] = dp_y[:end]

        padded_dp_maps = torch.cat([padded_dp_x,padded_dp_y], dim=1)
        return padded_dp_maps

    def dp_merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(
            len(sequences), max(lengths)
        ).fill_(config.UNK_DP).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(
            len(sequences), max(lengths)
        ).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["context"]), reverse=True)  ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    input_batch, input_lengths = merge(item_info["context"])
    mask_input, mask_input_lengths = merge(item_info["context_mask"])
    emotion_batch, emotion_lengths = merge(item_info["emotion_context"])
    input_rel_batch, input_rel_length = dp_merge(item_info["context_dp"])
    input_dp_map = dp_map_merge(item_info["context_dp_map"])

    ## target
    target_batch, target_lengths = merge(item_info["target"])
    target_rel_batch, target_rel_length = dp_merge(item_info["target_dp"])
    target_rel_f_batch, target_rel_f_length = dp_merge(item_info["target_dp_f"])
    target_dp_map = dp_map_merge(item_info["target_dp_map"])

    input_batch = input_batch.to(config.device)
    mask_input = mask_input.to(config.device)
    input_rel_batch = input_rel_batch.to(config.device)
    input_dp_map = input_dp_map.to(config.device)
    target_batch = target_batch.to(config.device)
    target_rel_batch = target_rel_batch.to(config.device)
    target_rel_f_batch = target_rel_f_batch.to(config.device)
    target_dp_map = target_dp_map.to(config.device)


    d = {}
    d["input_batch"] = input_batch
    d["input_lengths"] = torch.LongTensor(input_lengths)
    d["mask_input"] = mask_input
    d["target_batch"] = target_batch
    d["target_lengths"] = torch.LongTensor(target_lengths)
    d["emotion_context_batch"] = emotion_batch.to(config.device)

    d["input_rel_batch"] = input_rel_batch
    d["input_dp_map"] = input_dp_map
    d["target_rel_batch"] = target_rel_batch
    d["target_rel_f_batch"] = target_rel_f_batch
    d["target_dp_map"] = target_dp_map

    d["target_program"] = item_info["emotion"]
    d["program_label"] = item_info["emotion_label"]

    d["input_txt"] = item_info["context_text"]
    d["target_txt"] = item_info["target_text"]
    d["program_txt"] = item_info["emotion_text"]
    d["situation_txt"] = item_info["situation_text"]

    d["context_emotion_scores"] = item_info["context_emotion_scores"]

    return d


def prepare_data_seq(batch_size=32):
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()

    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    save_config()
    return (
        data_loader_tra,
        data_loader_val,
        data_loader_tst,
        vocab,
        len(dataset_train.emo_map),
    )
