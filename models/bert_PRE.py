from __future__ import absolute_import

import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from models.tcn import TemporalConvNet, TemporalConvNet_BN, TemporalConvNet_nochomp, TemporalConvNet_nochomp_BN
from models.module import *
import numpy as np
import copy
from collections import defaultdict
import models.PRE as PRE

__all__ = ['Transfromer_Baseline', 'Transfromer_Baseline_2class']

pretrain_model_dict = {
    'bert': BertModel,
    'roberta': BertModel,
}

pretrain_token_dict = {
    'bert': BertTokenizer,
    'roberta': BertTokenizer,
}

heads_no_need_permute = ['gru', 'bilstm']

heads_dict = {
    'res_block': ResBlock_Basic,
    'res_bottleneck_block': ResBlock_Bottleneck,
    'inception_block': Inception_block,
    'res_inception_block': Res_Inception_block,
    'tcn_nobn_chomp': TemporalConvNet,
    'tcn_bn_chomp': TemporalConvNet_BN,
    'tcn_nobn_nochomp': TemporalConvNet_nochomp,
    'tcn_bn_nochomp': TemporalConvNet_nochomp_BN,
    'gru': GRUBlock,
    'bilstm': BiLstmBlock,
}

entity_feat_module_dict = {
    'entity_avg_feats': entity_avg_feats,
    'entity_max_feats': entity_max_feats,
    'entity_avg_max_feats': entity_avg_max_feats,
    'entity_start_end_avg_max_feats': entity_start_end_avg_max_feats,
    'entity_avg_max_fc_bn_relu_feats': entity_avg_max_fc_bn_relu_feats,
    'entity_avg_max_fc_bn_feats': entity_avg_max_fc_bn_feats,
    'entity_avg_max_fc_bn_relu_feats_v2': entity_avg_max_fc_bn_relu_feats_v2,
    'entity_avg_max_fc_bn_feats_v2': entity_avg_max_fc_bn_feats_v2,
    'entity_avg_max_cls_feats': entity_avg_max_cls_feats,
    'entity_avg_max_globalmax_feats': entity_avg_max_globalmax_feats,
    'entity_avg_max_globalavg_feats': entity_avg_max_globalavg_feats,
    'entity_avg_max_globalavgmax_feats': entity_avg_max_globalavgmax_feats,
    'entity_avg_max_product_feats': entity_avg_max_product_feats,
}

feat_process_module_dict = {
    'fc_bn_relu': fc_bn_relu,

}


class Transfromer_Baseline(nn.Module):
    def __init__(self, args, **kwargs):
        super(Transfromer_Baseline, self).__init__()
        self.args = args
        self.max_predicate_num = 40
        self.pretrain_model_name = args.base_model.lower()
        self.tokenizer = pretrain_token_dict[self.pretrain_model_name].from_pretrained(
            args.model_path)
        base_model = pretrain_model_dict[self.pretrain_model_name].from_pretrained(
            args.model_path)
        self.base_model_embeddings = base_model.embeddings
        self.base_model_encoder = base_model.encoder
        # self.base_model_pooler = base_model.pooler
        # BertTokenizer.from_pretrained('bert-base-uncased')
        # BertModel.from_pretrained('bert-base-uncased')
        self.configuration = base_model.config

        del base_model

        if not args.head == '':
            self.heads = heads_dict[args.head](self.configuration.to_dict()[
                                               'hidden_size'], args.head_out_channels)
            self.classifier = nn.Linear(
                self.heads.out_channels, args.num_classes)
        else:
            self.classifier = nn.Linear(self.configuration.to_dict()[
                                        'hidden_size'], args.num_classes)

        self.entity_feat_module_name = args.entity_feat_module_name
        self.entity_feat_module = entity_feat_module_dict[self.entity_feat_module_name](
            self.configuration.to_dict()['hidden_size'])
        self.predicate_classifier = nn.Linear(self.configuration.to_dict(
        )['hidden_size'] * self.entity_feat_module.feats_num_coef, 25)

        self.sigmoid = nn.Sigmoid()
        self.relattention = PRE.MultiHeadAttentionLayer(
            self.configuration.to_dict()['hidden_size'])

        # if args.bool_crf:
        #     self.crf = CRF(args.num_classes * args.max_seq_length, batch_first=True)

    def entity_feats_fuse(self, subject_feats_in, object_feats_in):
        subject_feats = self.predicate_avgpool(
            subject_feats_in).permute(0, 2, 1)
        object_feats = self.predicate_avgpool(object_feats_in).permute(0, 2, 1)
        feats_fuse = torch.cat([subject_feats, object_feats], dim=2).squeeze(0)
        return feats_fuse

    # prob提取的向量，predicate_i当前关系的类别
    def find_entity_predicate_i(self, prob, predicate_i):
        predicate_i_label = predicate_i + 1  # add None
        subject_list, object_list = [], []
        spo_list = []
        subject_start = np.argwhere(prob[:, 0] == 1)[:, 0]
        subject_end = np.argwhere(prob[:, 1] == 1)[:, 0]
        object_start = np.argwhere(prob[:, 2] == 1)[:, 0]
        object_end = np.argwhere(prob[:, 3] == 1)[:, 0]

        for subject_start_i in subject_start:
            subject_end_i = subject_end[np.argwhere(
                subject_end >= subject_start_i)[:, 0]]
            if subject_end_i.shape[0] > 0:
                subject_list.append([subject_start_i, subject_end_i[0]])

        if len(subject_list) == 0:
            return []

        for object_start_i in object_start:
            object_end_i = object_end[np.argwhere(
                object_end >= object_start_i)[:, 0]]
            if object_end_i.shape[0] > 0:
                object_list.append([object_start_i, object_end_i[0]])

        if len(object_list) == 0:
            return []

        for subject_i in subject_list:  # 遍历所有的subject和object两两组合
            for object_i in object_list:
                spo_list.append([subject_i[0], subject_i[1],
                                object_i[0], object_i[1], predicate_i_label])

        return spo_list

    def find_entity_all(self, prob, seq_len):

        prob = prob[:, :, 1:]  # remove O
        num_predicate_class = prob.shape[2] // 4
        spo_list = []
        for batch_i in range(prob.shape[0]):  # 遍历每一个batch
            seq_len_batch_i = seq_len[batch_i]
            # only seq_len预测结果
            prob_i = prob[batch_i, 1: seq_len_batch_i + 1, :]
            spo_list_batch_i = []
            for predicate_i in range(num_predicate_class):
                prob_predicate_i = prob_i[:, [predicate_i, predicate_i+num_predicate_class,
                                              predicate_i+num_predicate_class*2, predicate_i+num_predicate_class*3]]
                spo_predicate_i = self.find_entity_predicate_i(
                    prob_predicate_i, predicate_i)
                if len(spo_predicate_i) > 0:
                    spo_list_batch_i.extend(spo_predicate_i)
            spo_list.append(spo_list_batch_i)
        return spo_list

    def spo_feats(self, feats, spo_list):
        batch = feats.shape[0]
        feats_spo = []
        feats_spo_num = []

        for batch_i in range(batch):
            spo_list_batch_i = spo_list[batch_i]

            for spo_index_batch_i in range(min(len(spo_list_batch_i), self.max_predicate_num)):
                spo_i_batch_i = spo_list_batch_i[spo_index_batch_i]
                subject_start_index = spo_i_batch_i[0] + 1  # add [CLS]
                subject_end_index = spo_i_batch_i[1] + 1
                object_start_index = spo_i_batch_i[2] + 1
                object_end_index = spo_i_batch_i[3] + 1
                feats_subject_i = feats[batch_i: batch_i+1,
                                        subject_start_index: subject_end_index + 1, :].permute(0, 2, 1)
                feats_object_i = feats[batch_i: batch_i+1,
                                       object_start_index: object_end_index + 1, :].permute(0, 2, 1)

                feats_spo_i = self.entity_feat_module(
                    feats_subject_i, feats_object_i)

                feats_spo.append(feats_spo_i)

            feats_spo_num.append(
                min(len(spo_list_batch_i), self.max_predicate_num))

        feats_spo = torch.cat(feats_spo, dim=0)
        return feats_spo, feats_spo_num

    def spo_labels(self, gt_spo_list, gt_predicate_num, pre_spo_list):
        batch = gt_spo_list.shape[0]
        labels_spo = []
        pre_gt_spo_list = []
        for batch_i in range(batch):
            pre_spo_list_batch_i = pre_spo_list[batch_i]
            gt_spo_list_batch_i = gt_spo_list[batch_i]
            pre_gt_spo_dict = defaultdict(dict)
            for pre_spo_index_batch_i in range(min(len(pre_spo_list_batch_i), self.max_predicate_num)):
                pre_spo_i_batch_i = pre_spo_list_batch_i[pre_spo_index_batch_i]

                pre_predicate_index = pre_spo_i_batch_i[4]

                pre_gt_spo_dict[str(pre_spo_index_batch_i)
                                ]['pre_subject_index'] = pre_spo_i_batch_i[0:2]
                pre_gt_spo_dict[str(pre_spo_index_batch_i)
                                ]['pre_object_index'] = pre_spo_i_batch_i[2:4]
                pre_gt_spo_dict[str(pre_spo_index_batch_i)
                                ]['pre_predicate'] = pre_spo_i_batch_i[4]
                pre_gt_spo_dict[str(pre_spo_index_batch_i)]['gt_predicate'] = 0

                bool_pre_corect = False
                for gt_spo_index_batch_i in range(gt_predicate_num[batch_i]):
                    gt_spo_i_batch_i = gt_spo_list_batch_i[gt_spo_index_batch_i]

                    gt_predicate_index = gt_spo_i_batch_i[4]

                    if pre_spo_i_batch_i == list(gt_spo_i_batch_i):
                        labels_spo.append(gt_predicate_index)
                        bool_pre_corect = True
                        pre_gt_spo_dict[str(
                            pre_spo_index_batch_i)]['gt_predicate'] = gt_predicate_index
                        break
                if not bool_pre_corect:
                    labels_spo.append(0)
            pre_gt_spo_list.append(pre_gt_spo_dict)

        labels_spo = torch.tensor(labels_spo).long()
        return labels_spo

    def forward(self, x):
        seq_lens = x['seq_lens']
        gt_predicate_labels, gt_predicate_num = x['predicate_labels'], x['predicate_num']
        x_embed = self.base_model_embeddings(x['input_ids'])
        x_encoder = self.base_model_encoder(x_embed)
        # x_pool = self.base_model_pooler(x_encoder[0])
        # x_encoder = self.base_model(**x)
        if not self.args.head == '':
            if self.args.head in heads_no_need_permute:
                x = self.heads(x_encoder.last_hidden_state)
                x = self.classifier(x)
            else:
                x = self.heads(x_encoder.last_hidden_state.permute(0, 2, 1))
                x = x.permute(0, 2, 1)
                x = self.classifier(x)
        else:
            # print(x_encoder.last_hidden_state.size())([20,128,768])
            pre = self.relattention(
                x_encoder.last_hidden_state, x_encoder.last_hidden_state, x_encoder.last_hidden_state,)
            x = self.classifier(x_encoder.last_hidden_state)

        with torch.no_grad():
            prob_x = self.sigmoid(x).cpu()
            prob_x_0_1 = torch.where(
                prob_x > 0.5, torch.tensor(1), torch.tensor(0)).numpy()

            pre_spo_list = self.find_entity_all(
                prob_x_0_1, seq_lens)  # 返回得到所有起始

        feats_spo, pre_spo_num = self.spo_feats(
            x_encoder.last_hidden_state, pre_spo_list)
        # print("feats_spo", feats_spo)

        predicate_pre = self.predicate_classifier(feats_spo)

        if self.training:
            labels_spo = self.spo_labels(
                gt_predicate_labels, gt_predicate_num, pre_spo_list)
            return x, predicate_pre, labels_spo, pre_spo_num, pre_spo_list
        else:
            return x, predicate_pre, pre_spo_num, pre_spo_list


class Transfromer_Baseline_2class(nn.Module):
    def __init__(self, args, **kwargs):
        super(Transfromer_Baseline_2class, self).__init__()
        self.args = args
        self.max_predicate_num = 40
        self.pretrain_model_name = args.base_model.lower()
        self.tokenizer = pretrain_token_dict[self.pretrain_model_name].from_pretrained(
            args.model_path)
        base_model = pretrain_model_dict[self.pretrain_model_name].from_pretrained(
            args.model_path)
        self.base_model_embeddings = base_model.embeddings
        self.base_model_encoder = base_model.encoder
        # self.base_model_pooler = base_model.pooler
        # BertTokenizer.from_pretrained('bert-base-uncased')
        # BertModel.from_pretrained('bert-base-uncased')
        self.configuration = base_model.config

        del base_model

        if not args.head == '':
            self.heads = heads_dict[args.head](self.configuration.to_dict()[
                                               'hidden_size'], args.head_out_channels)
            self.classifier = nn.Linear(
                self.heads.out_channels, args.num_classes)
        else:
            self.classifier = nn.Linear(self.configuration.to_dict()[
                                        'hidden_size'], args.num_classes)

        self.entity_feat_module_name = args.entity_feat_module_name
        self.entity_feat_module = entity_feat_module_dict[self.entity_feat_module_name](
            self.configuration.to_dict()['hidden_size'])
        feats_channels_num = self.configuration.to_dict(
        )['hidden_size'] * self.entity_feat_module.feats_num_coef

        self.feat_process_module = feat_process_module_dict[args.feat_process_module](
            feats_channels_num, args.num_fc)

        self.predicate_classifier = nn.Linear(feats_channels_num, 2)

        self.sigmoid = nn.Sigmoid()
        self.relattention = PRE.MultiHeadAttentionLayer(
            self.configuration.to_dict()['hidden_size'])
        self.dropout = nn.Dropout(0.1)
        self.self_attn_layer_norm = nn.LayerNorm(
            self.configuration.to_dict()['hidden_size'])
        # if args.bool_crf:
        #     self.crf = CRF(args.num_classes * args.max_seq_length, batch_first=True)

    def entity_feats_fuse(self, subject_feats_in, object_feats_in):
        subject_feats = self.predicate_avgpool(
            subject_feats_in).permute(0, 2, 1)
        object_feats = self.predicate_avgpool(object_feats_in).permute(0, 2, 1)
        feats_fuse = torch.cat([subject_feats, object_feats], dim=2).squeeze(0)
        return feats_fuse

    def find_entity_predicate_i(self, prob, predicate_i):
        predicate_i_label = predicate_i + 1  # add None
        subject_list, object_list = [], []
        spo_list = []
        subject_start = np.argwhere(prob[:, 0] == 1)[:, 0]
        subject_end = np.argwhere(prob[:, 1] == 1)[:, 0]
        object_start = np.argwhere(prob[:, 2] == 1)[:, 0]
        object_end = np.argwhere(prob[:, 3] == 1)[:, 0]

        for subject_start_i in subject_start:
            subject_end_i = subject_end[np.argwhere(
                subject_end >= subject_start_i)[:, 0]]
            if subject_end_i.shape[0] > 0:
                subject_list.append([subject_start_i, subject_end_i[0]])

        if len(subject_list) == 0:
            return []

        for object_start_i in object_start:
            object_end_i = object_end[np.argwhere(
                object_end >= object_start_i)[:, 0]]
            if object_end_i.shape[0] > 0:
                object_list.append([object_start_i, object_end_i[0]])

        if len(object_list) == 0:
            return []

        for subject_i in subject_list:
            for object_i in object_list:
                spo_list.append([subject_i[0], subject_i[1],
                                object_i[0], object_i[1], predicate_i_label])

        return spo_list

    def find_entity_all(self, prob, seq_len):

        prob = prob[:, :, 1:]  # remove O
        num_predicate_class = prob.shape[2] // 4
        spo_list = []
        for batch_i in range(prob.shape[0]):
            seq_len_batch_i = seq_len[batch_i]
            prob_i = prob[batch_i, 1: seq_len_batch_i + 1, :]  # only seq_len
            spo_list_batch_i = []
            for predicate_i in range(num_predicate_class):
                prob_predicate_i = prob_i[:, [predicate_i, predicate_i+num_predicate_class,
                                              predicate_i+num_predicate_class*2, predicate_i+num_predicate_class*3]]
                spo_predicate_i = self.find_entity_predicate_i(
                    prob_predicate_i, predicate_i)
                if len(spo_predicate_i) > 0:
                    spo_list_batch_i.extend(spo_predicate_i)
            spo_list.append(spo_list_batch_i)
        return spo_list

    def spo_feats(self, feats, spo_list):
        batch = feats.shape[0]
        feats_spo = []
        feats_spo_num = []

        for batch_i in range(batch):
            spo_list_batch_i = spo_list[batch_i]

            for spo_index_batch_i in range(min(len(spo_list_batch_i), self.max_predicate_num)):
                spo_i_batch_i = spo_list_batch_i[spo_index_batch_i]
                subject_start_index = spo_i_batch_i[0] + 1  # add [CLS]
                subject_end_index = spo_i_batch_i[1] + 1
                object_start_index = spo_i_batch_i[2] + 1
                object_end_index = spo_i_batch_i[3] + 1
                feats_subject_i = feats[batch_i: batch_i+1,
                                        subject_start_index: subject_end_index + 1, :].permute(0, 2, 1)
                feats_object_i = feats[batch_i: batch_i+1,
                                       object_start_index: object_end_index + 1, :].permute(0, 2, 1)

                feats_spo_i = self.entity_feat_module(
                    feats_subject_i, feats_object_i, feats[batch_i: batch_i+1])

                feats_spo.append(feats_spo_i)

            feats_spo_num.append(
                min(len(spo_list_batch_i), self.max_predicate_num))

        feats_spo = torch.cat(feats_spo, dim=0)
        return feats_spo, feats_spo_num

    def spo_labels(self, gt_spo_list, gt_predicate_num, pre_spo_list):
        batch = gt_spo_list.shape[0]
        labels_spo = []
        pre_gt_spo_list = []
        for batch_i in range(batch):
            pre_spo_list_batch_i = pre_spo_list[batch_i]
            gt_spo_list_batch_i = gt_spo_list[batch_i]
            pre_gt_spo_dict = defaultdict(dict)
            for pre_spo_index_batch_i in range(min(len(pre_spo_list_batch_i), self.max_predicate_num)):
                pre_spo_i_batch_i = pre_spo_list_batch_i[pre_spo_index_batch_i]

                pre_predicate_index = pre_spo_i_batch_i[4]

                pre_gt_spo_dict[str(pre_spo_index_batch_i)
                                ]['pre_subject_index'] = pre_spo_i_batch_i[0:2]
                pre_gt_spo_dict[str(pre_spo_index_batch_i)
                                ]['pre_object_index'] = pre_spo_i_batch_i[2:4]
                pre_gt_spo_dict[str(pre_spo_index_batch_i)
                                ]['pre_predicate'] = pre_spo_i_batch_i[4]
                pre_gt_spo_dict[str(pre_spo_index_batch_i)]['gt_predicate'] = 0

                bool_pre_corect = False
                for gt_spo_index_batch_i in range(gt_predicate_num[batch_i]):
                    gt_spo_i_batch_i = gt_spo_list_batch_i[gt_spo_index_batch_i]

                    gt_predicate_index = gt_spo_i_batch_i[4]

                    if pre_spo_i_batch_i == list(gt_spo_i_batch_i):
                        labels_spo.append(1)
                        bool_pre_corect = True
                        pre_gt_spo_dict[str(
                            pre_spo_index_batch_i)]['gt_predicate'] = gt_predicate_index
                        break
                if not bool_pre_corect:
                    labels_spo.append(0)
            pre_gt_spo_list.append(pre_gt_spo_dict)

        labels_spo = torch.tensor(labels_spo).long()
        return labels_spo

    def forward(self, x):
        seq_lens = x['seq_lens']
        gt_predicate_labels, gt_predicate_num = x['predicate_labels'], x['predicate_num']
        x_embed = self.base_model_embeddings(x['input_ids'])
        x_encoder = self.base_model_encoder(x_embed)
        # x_pool = self.base_model_pooler(x_encoder[0])
        # x_encoder = self.base_model(**x)
        if not self.args.head == '':
            if self.args.head in heads_no_need_permute:
                x = self.heads(x_encoder.last_hidden_state)
                x = self.classifier(x)
            else:
                x = self.heads(x_encoder.last_hidden_state.permute(0, 2, 1))
                x = x.permute(0, 2, 1)
                x = self.classifier(x)
        else:
           
            # print(x_encoder.last_hidden_state.size())([20,128,768])
            x = self.classifier(x_encoder.last_hidden_state)
        
        with torch.no_grad():
            prob_x = self.sigmoid(x).cpu()
            prob_x_0_1 = torch.where(
                prob_x > 0.5, torch.tensor(1), torch.tensor(0)).numpy()
            pre_spo_list = self.find_entity_all(prob_x_0_1, seq_lens)
        pre = self.relattention(
                x_encoder.last_hidden_state, x_encoder.last_hidden_state, x_encoder.last_hidden_state)
        src = self.self_attn_layer_norm(
                x_encoder.last_hidden_state + self.dropout(pre))
        feats_spo, pre_spo_num = self.spo_feats(
            src, pre_spo_list)

        if feats_spo.shape[0] <= 1:
            if self.training:
                return x, None, None, pre_spo_num, pre_spo_list

            elif feats_spo.shape[0] == 1:
                feats_spo = self.feat_process_module(feats_spo)
                predicate_pre = self.predicate_classifier(feats_spo)
                return x, predicate_pre, pre_spo_num, pre_spo_list
            else:
                return x, None, pre_spo_num, pre_spo_list
        
        feats_spo = self.feat_process_module(feats_spo)

        predicate_pre = self.predicate_classifier(feats_spo)

        if self.training:
            labels_spo = self.spo_labels(
                gt_predicate_labels, gt_predicate_num, pre_spo_list)

            return x, predicate_pre, labels_spo, pre_spo_num, pre_spo_list
        else:
            return x, predicate_pre, pre_spo_num, pre_spo_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default='Transfromer_Baseline_2class', type=str, help="model.")
    parser.add_argument("--base_model", default='roberta',
                        type=str, help="Pretrain model.")
    parser.add_argument("--entity_feat_module_name",
                        default='entity_avg_max_globalavgmax_feats', type=str, help="Pretrain model.")
    parser.add_argument("--feat_process_module",
                        default='fc_bn_relu', type=str, help="Pretrain model.")
    parser.add_argument("--num_fc", default=2, type=int,
                        help="Pretrain model.")
    parser.add_argument("--model_path", default='/home/yangzhenyu/Duie_torch_baseline/pretrain_model/bertbase',
                        type=str, required=False, help="Path to data.")
    parser.add_argument("--do_lower_case", action='store_true',
                        default=False, help="whether lower_case")
    parser.add_argument("--num_classes", type=int,
                        default=97, help="classification classes")
    parser.add_argument("--head", type=str, default='',
                        help="classification classes")
    parser.add_argument("--head_out_channels", type=int,
                        nargs='+', default=256, help="classification classes")
    parser.add_argument("--bool_crf", action='store_true',
                        default=False, help="whether lower_case")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = Transfromer_Baseline_2class(args)
    model_state_dict = torch.load(
        '../log/0421/Transfromer_bert/star_end_bertbase/model_best/model_best_epoch_44_f1_88.28.pth')
    model.load_state_dict(model_state_dict, strict=False)

    tokenizer = model.tokenizer
    model = nn.DataParallel(model).cuda()
    inputs = tokenizer(
        "In Queens , North Shore Towers , near the Nassau border , supplanted a golf course , and housing replaced a gravel quarry in Douglaston .", return_tensors="pt")
    inputs['seq_lens'] = [inputs['input_ids'].shape[1] - 2] * 2
    inputs['input_ids'] = (torch.cat([inputs['input_ids'], torch.zeros(
        [1, 128]).long()], dim=1)[:, 0: 128]).repeat(2, 1).cuda()
    predicate_labels = [[0] * 5] * 40
    predicate_labels[0] = [26, 27, 1, 1, 6]
    predicate_labels = np.array(predicate_labels)[np.newaxis, :].repeat(2, 0)
    inputs['predicate_labels'] = predicate_labels
    inputs['predicate_num'] = np.array([1, 1])
    # print(inputs)

    # bert_model = BertModel.from_pretrained('/home/yangzhenyu/DuIE_torch_wangzhen/pretrain_model/chinese_roberta_wwm_ext_pytorch')
    # out_bert = bert_model(**inputs)
    out, predicate_pre, labels_spo, feats_spo_num, pre_spo_list = model(inputs)
    # torch.save(model.state_dict(), './model.pth')
    print(out.shape)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(predicate_pre, labels_spo.cuda())
