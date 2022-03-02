from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import collections
import json
import os.path as osp
from scipy.io import loadmat
import numpy as np
from typing import Optional, List, Union, Dict
from tqdm import tqdm
from utils import Logger
import torch
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
from extract_chinese_and_punct import ChineseAndPunctuationExtractor
from transformers import RobertaTokenizer, AutoTokenizer, BertTokenizer

InputFeature = collections.namedtuple("InputFeature", [
    "input_ids", "seq_len", "tok_to_orig_start_index", "tok_to_orig_end_index",
    "labels", "predicate_labels", "predicate_num"
])

def parse_label(spo_list, label_map, tokens, tokenizer):
    # 2 tags for each predicate + I tag + O tag
    max_predicate_num = 40
    valid_predicate_num = len(label_map.keys()) - 1
    num_labels = 4 * valid_predicate_num + 1
    seq_len = len(tokens)
    # print(tokens)
    # initialize tag
    labels = [[0] * num_labels for i in range(seq_len)]

    spo_predicate_info = {}
    spo_index = 0
    total_valid_predicate_num = 0
    predicate_labels = []
    for i in range(max_predicate_num):
        predicate_labels.append([0,0,0,0,0])
    #  find all entities and tag them with corresponding "B"/"I" labels
    for spo in spo_list:
        # for spo_object in spo['object'].keys():
        #     # assign relation label
            if spo['label'] in label_map.keys():
                # simple relation
                label_subject = label_map[spo['label']]
                label_object = label_subject + valid_predicate_num * 2
                subject_tokens = tokenizer._tokenize(spo['em1Text'])
                object_tokens = tokenizer._tokenize(spo['em2Text'])
                label_relation = spo['label']
                label_relation_label = label_map[label_relation]
            # else:
            #     # complex relation
            #     label_subject = label_map[spo['predicate'] + '_' + spo_object]
            #     label_object = label_subject + 55
            #     subject_tokens = tokenizer._tokenize(spo['subject'])
            #     object_tokens = tokenizer._tokenize(spo['object'][spo_object])

            subject_tokens_len = len(subject_tokens)
            object_tokens_len = len(object_tokens)

            # assign token label
            # there are situations where s entity and o entity might overlap, e.g. xyz established xyz corporation
            # to prevent single token from being labeled into two different entity
            # we tag the longer entity first, then match the shorter entity within the rest text
            forbidden_index = None
            label_dict = {'predicate': label_relation, 'predicate_label': label_relation_label}
            if subject_tokens_len > object_tokens_len:
                for index in range(seq_len - subject_tokens_len + 1):
                    if tokens[index: index + subject_tokens_len] == subject_tokens:
                        labels[index][label_subject] = 1
                        for i in range(subject_tokens_len - 2):
                            labels[index + i + 1][0] = 1
                        labels[index + subject_tokens_len - 1][label_subject + valid_predicate_num] = 1
                        forbidden_index = index
                        label_dict['subject'] = spo['em1Text']
                        label_dict['subject_index'] = [index, index + subject_tokens_len - 1]
                        label_dict['subject_length'] = subject_tokens_len
                        break

                for index in range(seq_len - object_tokens_len + 1):
                    if tokens[index:index + object_tokens_len] == object_tokens:
                        if forbidden_index is None:
                            labels[index][label_object] = 1
                            for i in range(object_tokens_len - 2):
                                labels[index + i + 1][0] = 1
                            labels[index + object_tokens_len - 1][label_object + valid_predicate_num] = 1
                            label_dict['object'] = spo['em2Text']
                            label_dict['object_index'] = [index, index + object_tokens_len - 1]
                            label_dict['object_length'] = object_tokens_len
                            break
                        # check if labeled already
                        elif index < forbidden_index or index >= forbidden_index + len(subject_tokens):
                            labels[index][label_object] = 1
                            for i in range(object_tokens_len - 2):
                                labels[index + i + 1][0] = 1
                            labels[index + object_tokens_len - 1][label_object + valid_predicate_num] = 1
                            label_dict['object'] = spo['em2Text']
                            label_dict['object_index'] = [index, index + object_tokens_len - 1]
                            label_dict['object_length'] = object_tokens_len
                            break

            else:
                for index in range(seq_len - object_tokens_len + 1):
                    if tokens[index:index + object_tokens_len] == object_tokens:
                        labels[index][label_object] = 1
                        for i in range(object_tokens_len - 2):
                            labels[index + i + 1][0] = 1
                        labels[index + object_tokens_len - 1][label_object + valid_predicate_num] = 1
                        forbidden_index = index
                        label_dict['object'] = spo['em2Text']
                        label_dict['object_index'] = [index, index + object_tokens_len - 1]
                        label_dict['object_length'] = object_tokens_len
                        break

                for index in range(seq_len - subject_tokens_len + 1):
                    if tokens[index:index + subject_tokens_len] == subject_tokens:
                        if forbidden_index is None:
                            labels[index][label_subject] = 1
                            for i in range(subject_tokens_len - 2):
                                labels[index + i + 1][0] = 1
                            labels[index + subject_tokens_len - 1][label_subject + valid_predicate_num] = 1
                            label_dict['subject'] = spo['em1Text']
                            label_dict['subject_index'] = [index, index + subject_tokens_len - 1]
                            label_dict['subject_length'] = subject_tokens_len
                            break
                        elif index < forbidden_index or index >= forbidden_index + len( object_tokens):
                            labels[index][label_subject] = 1
                            for i in range(subject_tokens_len - 2):
                                labels[index + i + 1][0] = 1
                            labels[index + subject_tokens_len - 1][label_subject + valid_predicate_num] = 1
                            label_dict['subject'] = spo['em1Text']
                            label_dict['subject_index'] = [index, index + subject_tokens_len - 1]
                            label_dict['subject_length'] = subject_tokens_len
                            break
            if 'subject' in label_dict.keys() and 'object' in label_dict.keys():
                try:
                    predicate_labels[total_valid_predicate_num][0] = label_dict['subject_index'][0]
                    predicate_labels[total_valid_predicate_num][1] = label_dict['subject_index'][1]
                    predicate_labels[total_valid_predicate_num][2] = label_dict['object_index'][0]
                    predicate_labels[total_valid_predicate_num][3] = label_dict['object_index'][1]
                    predicate_labels[total_valid_predicate_num][4] = label_dict['predicate_label']

                    spo_predicate_info[str(total_valid_predicate_num)] = label_dict
                    total_valid_predicate_num += 1
                    label_dict = {}
                except:
                    import pdb
                    pdb.set_trace()
    # if token wasn't assigned as any "B"/"I" tag, give it an "O" tag for outside
    for i in range(seq_len):
        if labels[i] == [0] * num_labels:
            labels[i][0] = 1

    spo_predicate_info['total_valid_predicate_num'] = total_valid_predicate_num


    return labels, predicate_labels, spo_predicate_info


def convert_example_to_feature(example, tokenizer, label_map, max_length, pad_to_max_length):

    spo_list = example['relationMentions'] if "relationMentions" in example.keys(
    ) else None
    text_raw = example['sentText']
    
    sub_text = []
    cutwords1 = word_tokenize(text_raw)
    interpunctuations = [',', '.', ':', ';', '?',
                         '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']  # 瀹氫箟绗﹀彿鍒楄〃
    cutwords2 = [word for word in cutwords1 if word not in interpunctuations]
    stops = set(stopwords.words("english"))
    sub_text = [word for word in cutwords2 if word not in stops]

    tok_to_orig_start_index = []
    tok_to_orig_end_index = []
    orig_to_tok_index = []
    tokens = []
    text_tmp = ''
    for (i, token) in enumerate(sub_text):
        orig_to_tok_index.append(len(tokens))
        sub_tokens = tokenizer._tokenize(token)
        text_tmp += token
        for sub_token in sub_tokens:
            tok_to_orig_start_index.append(len(text_tmp) - len(token))
            tok_to_orig_end_index.append(len(text_tmp) - 1)
            tokens.append(sub_token)
            if len(tokens) >= max_length - 2:
                break
        else:
            continue
        break

    seq_len = len(tokens)
    # 2 tags for each predicate + I tag + O tag
    num_labels = 4 * (len(label_map.keys()) - 1) + 1
    # initialize tag
    labels = [[0] * num_labels for i in range(seq_len)]
    if spo_list is not None:
        labels, predicate_labels, spo_predicate_info = parse_label(spo_list, label_map, tokens, tokenizer)

    # add [CLS] and [SEP] token, they are tagged into "O" for outside
    if seq_len > max_length - 2:
        tokens = tokens[0:(max_length - 2)]
        labels = labels[0:(max_length - 2)]
        tok_to_orig_start_index = tok_to_orig_start_index[0:(max_length - 2)]
        tok_to_orig_end_index = tok_to_orig_end_index[0:(max_length - 2)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    # "O" tag for [PAD], [CLS], [SEP] token
    outside_label = [[1] + [0] * (num_labels - 1)]

    labels = outside_label + labels + outside_label
    tok_to_orig_start_index = [-1] + tok_to_orig_start_index + [-1]
    tok_to_orig_end_index = [-1] + tok_to_orig_end_index + [-1]
    if seq_len < max_length:
        tokens = tokens + ["[PAD]"] * (max_length - seq_len - 2)
        labels = labels + outside_label * (max_length - len(labels))
        tok_to_orig_start_index = tok_to_orig_start_index + [-1] * (
            max_length - len(tok_to_orig_start_index))
        tok_to_orig_end_index = tok_to_orig_end_index + [-1] * (
            max_length - len(tok_to_orig_end_index))

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return InputFeature(
        input_ids=np.array(token_ids),
        seq_len=np.array(seq_len),
        tok_to_orig_start_index=np.array(tok_to_orig_start_index),
        tok_to_orig_end_index=np.array(tok_to_orig_end_index),
        labels=np.array(labels),
        predicate_labels=np.array(predicate_labels), 
        predicate_num=np.array(spo_predicate_info['total_valid_predicate_num'])), spo_predicate_info


class Duie2_SPO(object):


    def __init__(self, 
                file_path: Union[str, os.PathLike],
                tokenizer: RobertaTokenizer,
                max_length: Optional[int]=512,
                pad_to_max_length: Optional[bool]=None):

        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length

        self.dataset = self._process_data(file_path, tokenizer, max_length, pad_to_max_length)

        self.input_ids = self.dataset.input_ids
        self.seq_len = self.dataset.seq_len
        self.tok_to_orig_start_index = self.dataset.tok_to_orig_start_index
        self.tok_to_orig_end_index = self.dataset.tok_to_orig_end_index
        self.labels = self.dataset.labels

        self.length = self.dataset.labels.shape[0]

        print('Total valid predicate num: {}'.format(self.total_valid_predicate_num))

    def __len__(self):
        return self.length

    def _process_data(self, 
                file_path: Union[str, os.PathLike],
                tokenizer: RobertaTokenizer,
                max_length: Optional[int]=512,
                pad_to_max_length: Optional[bool]=None):

        assert os.path.exists(file_path) and os.path.isfile(
            file_path), f"{file_path} dose not exists or is not a file."
        label_map_path = os.path.join(
            os.path.dirname(file_path), "rel2idstarend.json")
        assert os.path.exists(label_map_path) and os.path.isfile(
            label_map_path
        ), f"{label_map_path} dose not exists or is not a file."
        with open(label_map_path, 'r', encoding='utf8') as fp:
            label_map = json.load(fp)
        # chineseandpunctuationextractor = ChineseAndPunctuationExtractor()

        input_ids, seq_len, tok_to_orig_start_index, tok_to_orig_end_index, labels, predicate_labels, predicate_num = (
            [] for _ in range(7))
        dataset_scale = sum(1 for line in open(file_path, 'r'))
        self.total_valid_predicate_num = 0
        print("Preprocessing data, loaded from %s" % file_path)
        with open(file_path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in tqdm(lines):
                example = json.loads(line)
                input_feature, spo_predicate_info = convert_example_to_feature(
                    example, tokenizer,
                    label_map, max_length, pad_to_max_length)
                input_ids.append(input_feature.input_ids)
                seq_len.append(input_feature.seq_len)
                tok_to_orig_start_index.append(
                    input_feature.tok_to_orig_start_index)
                tok_to_orig_end_index.append(
                    input_feature.tok_to_orig_end_index)
                labels.append(input_feature.labels)
                predicate_labels.append(input_feature.predicate_labels)
                predicate_num.append(input_feature.predicate_num)

                self.total_valid_predicate_num += spo_predicate_info['total_valid_predicate_num']
        dataset= InputFeature(
            input_ids=np.array(input_ids),
            seq_len=np.array(seq_len),
            tok_to_orig_start_index=np.array(tok_to_orig_start_index),
            tok_to_orig_end_index=np.array(tok_to_orig_end_index),
            labels=np.array(labels), 
            predicate_labels=np.array(predicate_labels), 
            predicate_num=np.array(predicate_num), )
            
        return dataset


"""Create dataset"""

__factory = {
    'duie_spo': Duie2_SPO,
    # 'duie_spo_ensemble': Duie2_SPO_ensemble,
    # 'duie_spo_multilabel': Duie2_SPO_MultiLable,
}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](*args, **kwargs)

if __name__ == '__main__':
    # nltk.download('punkt')
    tokenizer = BertTokenizer.from_pretrained(
        '/home/yangzhenyu/Duie_torch_baseline/pretrain_model/bertbase', do_lower_case=False, cache_dir=None, use_fast=False)
    duie_dataset = Duie2_SPO('./data/webNLG/dev.json', tokenizer, 256, True)
    print(duie_dataset.length)





