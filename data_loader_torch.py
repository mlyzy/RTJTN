from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import random
import copy

from extract_chinese_and_punct import ChineseAndPunctuationExtractor
from transformers import RobertaTokenizer, AutoTokenizer, BertTokenizer
from torch.utils.data import DataLoader
from data_manager import *
class Duie_loader(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset[0].shape[0]

    def __getitem__(self, index):

        input_ids = torch.from_numpy(self.dataset[0][index])
        seq_lens = torch.tensor(self.dataset[1][index])
        tok_to_orig_start_index = torch.from_numpy(self.dataset[2][index])
        tok_to_orig_end_index = torch.from_numpy(self.dataset[3][index])
        labels = torch.from_numpy(self.dataset[4][index]).float()


        return (input_ids, seq_lens, tok_to_orig_start_index,
                tok_to_orig_end_index, labels)

class Duie_loader_predicate(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset[0].shape[0]

    def __getitem__(self, index):

        input_ids = torch.from_numpy(self.dataset[0][index])
        seq_lens = torch.tensor(self.dataset[1][index])
        tok_to_orig_start_index = torch.from_numpy(self.dataset[2][index])
        tok_to_orig_end_index = torch.from_numpy(self.dataset[3][index])
        labels = torch.from_numpy(self.dataset[4][index]).float()
        predicate_labels = torch.from_numpy(self.dataset[5][index])
        predicate_num = torch.tensor(self.dataset[6][index])


        return (input_ids, seq_lens, tok_to_orig_start_index,
                tok_to_orig_end_index, labels, predicate_labels, predicate_num)

# class Duie_SPO_P_recog_loader(Dataset):

#     def __init__(self, dataset, max_length = 156, mode = 'train'):
#         self.dataset = dataset
#         self.token_id_dict = {'[PAD]': 0, '[CLS]':101, '[SEP]': 102}
#         self.max_length = max_length
#         self.mode = mode

#         self.tokens_list = self.dataset.dataset.tokens_list
#         self.token_ids_list = self.dataset.dataset.token_ids
#         self.spo_list = self.dataset.dataset.spo_list
#         self.spo_seq_dict = self.dataset.spo_seq_dict
#         self.dataset_mode = self.dataset.mode

#     def __len__(self):
#         return len(self.spo_seq_dict.keys())

#     def __getitem__(self, index):
#         if self.dataset_mode == 'spo_base':
#             seq_index = int(self.spo_seq_dict[str(index)].split('_')[1])
#             spo_index = self.spo_seq_dict[str(index)].split('_')[2]
#             tokens = self.tokens_list[seq_index]
#             token_ids = self.token_ids_list[seq_index]
#             spo_info = self.spo_list[seq_index]
#             spo_predicate_info = spo_info[-1]
#             spo_select_info = spo_predicate_info[spo_index]
#         else:
#             seq_index = int(self.spo_seq_dict[str(index)].split('_')[1])
#             tokens = self.tokens_list[seq_index]
#             token_ids = self.token_ids_list[seq_index]
#             spo_info = self.spo_list[seq_index]
#             spo_predicate_info = spo_info[-1]
#             total_valid_predicate_num = spo_predicate_info['total_valid_predicate_num']
#             try:
#                 predicate_index_select = random.randint(0, total_valid_predicate_num - 1)
#                 spo_select_info = spo_predicate_info[str(predicate_index_select)]
#             except:
#                 spo_select_info = spo_predicate_info['0']

#         tokens += [''] * (self.max_length - len(tokens))

#         subject_index = copy.deepcopy(spo_select_info['subject_index'])
#         subject_length = copy.deepcopy(spo_select_info['subject_length'])
#         object_index = copy.deepcopy(spo_select_info['object_index'])
#         object_length = copy.deepcopy(spo_select_info['object_length'])
#         label = copy.deepcopy(spo_select_info['predicate_label'])

#         if self.mode == 'train' and random.random() < 0 and label != 0:
#             random_prob = random.random()
#             if random_prob < 0.4:# subject wrong
#                 label, subject_index, subject_length = self.ramdom_crop_pad(subject_index, subject_length, token_ids, label)

#             elif random_prob < 0.8:# object wrong
#                 label, object_index, object_length = self.ramdom_crop_pad(object_index, object_length, token_ids, label)

#             else:# subject and object wrong
#                 label, subject_index, subject_length = self.ramdom_crop_pad(subject_index, subject_length, token_ids, label)

#                 label, object_index, object_length = self.ramdom_crop_pad(object_index, object_length, token_ids, label)


#         if len(token_ids) + subject_length + object_length + 4 <= self.max_length:
#             token_ids_add_sub_obj = [101] + token_ids[subject_index[0]: subject_index[1]] + [102] + token_ids[object_index[0]: object_index[1]] + [102]
#             token_ids_add_sub_obj += token_ids + [102]
#             seq_lens = len(token_ids_add_sub_obj) - 2
#             token_ids_add_sub_obj += [0] * (self.max_length - len(token_ids_add_sub_obj))

#         elif subject_length + object_length + max(subject_index[1], object_index[1]) + 4 <= self.max_length:
#             token_ids_add_sub_obj = [101] + token_ids[subject_index[0]: subject_index[1]] + [102] + token_ids[object_index[0]: object_index[1]] + [102]
#             token_ids_add_sub_obj += token_ids[0: self.max_length - len(token_ids_add_sub_obj) - 1] + [102]
#             seq_lens = self.max_length - 2

#         elif subject_length + object_length + (max(subject_index[1], object_index[1]) - min(subject_index[0], object_index[0]))+ 4 <= self.max_length:
#             token_ids_add_sub_obj = [101] + token_ids[subject_index[0]: subject_index[1]] + [102] + token_ids[object_index[0]: object_index[1]] + [102]
#             token_ids_add_sub_obj += token_ids[min(subject_index[0], object_index[0]) : max(subject_index[1], object_index[1])] + [102]
#             seq_lens = len(token_ids_add_sub_obj) - 2
#             token_ids_add_sub_obj += [0] * (self.max_length - len(token_ids_add_sub_obj))
#         else:
#             subject_index = copy.deepcopy(spo_select_info['subject_index'])
#             subject_length = copy.deepcopy(spo_select_info['subject_length'])
#             object_index = copy.deepcopy(spo_select_info['object_index'])
#             object_length = copy.deepcopy(spo_select_info['object_length'])
#             label = copy.deepcopy(spo_select_info['predicate_label'])
#             if len(token_ids) + subject_length + object_length + 4 <= self.max_length:
#                 token_ids_add_sub_obj = [101] + token_ids[subject_index[0]: subject_index[1]] + [102] + token_ids[object_index[0]: object_index[1]] + [102]
#                 token_ids_add_sub_obj += token_ids + [102]
#                 seq_lens = len(token_ids_add_sub_obj) - 2
#                 token_ids_add_sub_obj += [0] * (self.max_length - len(token_ids_add_sub_obj))

#             elif subject_length + object_length + max(subject_index[1], object_index[1]) + 4 <= self.max_length:
#                 token_ids_add_sub_obj = [101] + token_ids[subject_index[0]: subject_index[1]] + [102] + token_ids[object_index[0]: object_index[1]] + [102]
#                 token_ids_add_sub_obj += token_ids[0: self.max_length - len(token_ids_add_sub_obj) - 1] + [102]
#                 seq_lens = self.max_length - 2

#             elif subject_length + object_length + (max(subject_index[1], object_index[1]) - min(subject_index[0], object_index[0]))+ 4 <= self.max_length:
#                 token_ids_add_sub_obj = [101] + token_ids[subject_index[0]: subject_index[1]] + [102] + token_ids[object_index[0]: object_index[1]] + [102]
#                 token_ids_add_sub_obj += token_ids[min(subject_index[0], object_index[0]) : max(subject_index[1], object_index[1])] + [102]
#                 seq_lens = len(token_ids_add_sub_obj) - 2
#                 token_ids_add_sub_obj += [0] * (self.max_length - len(token_ids_add_sub_obj))
#             else:
#                 token_ids_add_sub_obj = [101] + [102] + [102] + token_ids + [102]
#                 seq_lens = len(token_ids_add_sub_obj) - 2
#                 token_ids_add_sub_obj += [0] * (self.max_length - len(token_ids_add_sub_obj))
#                 subject_index = [0, 0]
#                 object_index = [0, 0]
#                 subject_length = object_length = 0
#                 label = 0
#         # if len(token_ids_add_sub_obj) != self.max_length or subject_index[1] > len(token_ids) or object_index[1] > len(token_ids):
#         #     import pdb
#         #     pdb.set_trace()
#         return (torch.tensor(token_ids_add_sub_obj), torch.tensor(seq_lens), torch.tensor(label), tokens, torch.tensor(subject_index), torch.tensor(object_index))
    
#     def ramdom_crop_pad(self, subject_index, subject_length, token_ids, label):
#         subject_index_random = [0, 0]
#         random_prob = random.random()
#         try:
#             if random_prob < 0.4:# subject right less: 绁為洉渚?#                 if subject_length > 1:
#                     subject_length_random = random.randint(1, subject_length -1)
#                     subject_index_random[0] = subject_index[0]
#                     subject_index_random[1] = subject_index_random[0] + subject_length_random
#                 else:
#                     subject_length_random = 1
#                     subject_index_random[0] = random.randint(0, len(token_ids) - 1)
#                     while (subject_index_random[0] == subject_index[0]):
#                         subject_index_random[0] = random.randint(0, len(token_ids) - 1)
#                     subject_index_random[1] = subject_index_random[0] + 1
#                 # if subject_index_random[1] > len(token_ids) or subject_index_random[0] < 0:
#                 #     import pdb
#                 #     pdb.set_trace()
#             elif random_prob < 0.5:# subject left less: 绁為洉渚?#                 if subject_length > 1:
#                     subject_length_random = random.randint(1, subject_length -1)
#                     subject_index_random[1] = subject_index[1]
#                     subject_index_random[0] = subject_index_random[1] - subject_length_random
#                 else:
#                     subject_length_random = 1
#                     subject_index_random[0] = random.randint(0, len(token_ids) - 1)
#                     while (subject_index_random[0] == subject_index[0]):
#                         subject_index_random[0] = random.randint(0, len(token_ids) - 1)
#                     subject_index_random[1] = subject_index_random[0] + 1
#                 # if subject_index_random[1] > len(token_ids) or subject_index_random[0] < 0:
#                 #     import pdb
#                 #     pdb.set_trace()
#             elif random_prob < 0.6:# subject left right less: 闆曚緺
#                 if subject_length >= 4:
#                     subject_sample_random = sorted(random.sample(list(range(subject_index[0] + 1, subject_index[1])), 2))
#                     subject_index_random[0] = subject_sample_random[0]
#                     subject_index_random[1] = subject_sample_random[1]
#                 else:
#                     subject_index_random = subject_index
#                 # if subject_index_random[1] > len(token_ids) or subject_index_random[0] < 0:
#                 #     import pdb
#                 #     pdb.set_trace()
#             elif random_prob < 0.7:#subject right more: 绁為洉渚犱荆鐢佃
#                 right_random_pad = random.randint(1, 4)
#                 subject_length = min(subject_index[1] + right_random_pad, len(token_ids)) - subject_index[0]
#                 subject_index_random[0] =  subject_index[0]
#                 subject_index_random[1] = subject_index_random[0] + subject_length
#                 # if subject_index_random[1] > len(token_ids) or subject_index_random[0] < 0:
#                 #     import pdb
#                 #     pdb.set_trace()
#             elif random_prob < 0.8:#subject left more: 鎾斁绁為洉渚犱荆
#                 left_random_pad = random.randint(1, 4)
#                 subject_length = subject_index[1] - max(subject_index[0] - left_random_pad, 0)
#                 subject_index_random[1] =  subject_index[1]
#                 subject_index_random[0] = subject_index_random[1] - subject_length
#                 # if subject_index_random[1] > len(token_ids) or subject_index_random[0] < 0:
#                 #     import pdb
#                 #     pdb.set_trace()
#             elif random_prob < 1.0:#subject left right more: 鎾斁绁為洉渚犱荆鐢佃
#                 left_random_pad = random.randint(1, 4)
#                 subject_length = subject_index[1] - max(subject_index[0] - left_random_pad, 0)
#                 subject_index_random[1] =  subject_index[1]
#                 subject_index_random[0] = subject_index_random[1] - subject_length
#                 right_random_pad = random.randint(1, 4)
#                 subject_length = min(subject_index_random[1] + right_random_pad, len(token_ids)) - subject_index_random[0]
#                 subject_index_random[1] = subject_index_random[0] + subject_length
#                 # if subject_index_random[1] > len(token_ids) or subject_index_random[0] < 0:
#                 #     import pdb
#                 #     pdb.set_trace()

#             subject_length_random = subject_index_random[1] - subject_index_random[0]
#         except:
#             # import pdb
#             # pdb.set_trace()
#             subject_index_random = subject_index
#             subject_length_random = subject_length
#         if subject_index_random == subject_index:
#             return label, subject_index_random, subject_length_random
#         else:
#             return 0, subject_index_random, subject_length_random

if __name__ == '__main__':

    random.seed(42)
    np.random.seed(42)
    tokenizer = BertTokenizer.from_pretrained('/home/yangzhenyu/Duie_torch_baseline/pretrain_model/bertbase', do_lower_case=False, cache_dir=None, use_fast=False)
    duie_dataset = Duie2_SPO('./data/NYT/test.json', tokenizer, 256, True)
    print(duie_dataset.length)
    trainloader = DataLoader(
        Duie_loader_predicate(duie_dataset.dataset),
        batch_size=16, 
        num_workers=0,
        pin_memory=False, 
        drop_last=True,
        shuffle=False,
    )
    for epoch in range(1):
        for step, batch in enumerate(trainloader):
            input_ids, seq_lens, tok_to_orig_start_index, tok_to_orig_end_index, labels, predicate_labels = batch
            input_ids = input_ids.cuda()

