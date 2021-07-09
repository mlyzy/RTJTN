from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class ResBlock_Basic(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlock_Basic, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(inplanes)

        self.out_channels = inplanes

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class ResBlock_Bottleneck(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlock_Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
    
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.conv3 = nn.Conv1d(planes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(inplanes)

        self.relu = nn.LeakyReLU(0.1)

        self.out_channels = inplanes

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()   

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)        

        out += residual
        out = self.relu(out)

        return out

class conv_bn_relu_1d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, padding=0):
        super(conv_bn_relu_1d, self).__init__()
        self.conv = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(planes)
        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class conv_bn_1d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, padding=0):
        super(conv_bn_1d, self).__init__()
        self.conv = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x

class Inception_block(nn.Module):

    def __init__(self, inplanes, planes):
        super(Inception_block, self).__init__()
        self.branch1_1 = conv_bn_relu_1d(inplanes, inplanes // 4, kernel_size=1)

        self.branch2_1 = conv_bn_relu_1d(inplanes, inplanes // 4, kernel_size=1)
        self.branch2_2_3 = conv_bn_relu_1d(inplanes // 4, inplanes // 4, kernel_size=3, padding=1)

        self.branch3_1 = conv_bn_relu_1d(inplanes, inplanes // 4, kernel_size=1)
        self.branch3_2_3 = conv_bn_relu_1d(inplanes // 4, inplanes // 4, kernel_size=3, padding=1)
        self.branch3_2_3 = conv_bn_relu_1d(inplanes // 4, inplanes // 4, kernel_size=3, padding=1)

        self.branch4_1 = conv_bn_relu_1d(inplanes, inplanes // 4, kernel_size=1)
        self.branch4_2_3 = conv_bn_relu_1d(inplanes // 4, inplanes // 4, kernel_size=3, padding=1)
        self.branch4_3_3 = conv_bn_relu_1d(inplanes // 4, inplanes // 4, kernel_size=3, padding=1)
        self.branch4_4_3 = conv_bn_relu_1d(inplanes // 4, inplanes // 4, kernel_size=3, padding=1)

        self.out_channels = inplanes // 4 * 4

    def forward(self, x):
        branch1 = self.branch1_1(x)

        branch2 = self.branch2_2_3(self.branch2_1(x))

        branch3 = self.branch3_2_3(self.branch3_2_3(self.branch3_1(x)))

        branch4 = self.branch4_4_3(self.branch4_3_3(self.branch4_2_3(self.branch4_1(x))))

        return torch.cat([branch1, branch2, branch3, branch4] , dim=1)

class Res_Inception_block(nn.Module):

    def __init__(self, inplanes, out):
        super(Res_Inception_block, self).__init__()
        assert (inplanes % 4) == 0
        planes = inplanes // 4
        self.branch1_1 = conv_bn_1d(inplanes, planes, kernel_size=1)

        self.branch2_1 = conv_bn_relu_1d(inplanes, planes, kernel_size=1)
        self.branch2_2_3 = conv_bn_1d(planes, planes, kernel_size=3, padding=1)

        self.branch3_1 = conv_bn_relu_1d(inplanes, planes, kernel_size=1)
        self.branch3_2_3 = conv_bn_relu_1d(planes, planes, kernel_size=3, padding=1)
        self.branch3_2_3 = conv_bn_1d(planes, planes, kernel_size=3, padding=1)

        self.branch4_1 = conv_bn_relu_1d(inplanes, planes, kernel_size=1)
        self.branch4_2_3 = conv_bn_relu_1d(planes, planes, kernel_size=3, padding=1)
        self.branch4_3_3 = conv_bn_relu_1d(planes, planes, kernel_size=3, padding=1)
        self.branch4_4_3 = conv_bn_1d(planes, planes, kernel_size=3, padding=1)

        self.relu = nn.LeakyReLU(0.1)

        self.out_channels = inplanes

    def forward(self, x):
        residual = x

        branch1 = self.branch1_1(x)

        branch2 = self.branch2_2_3(self.branch2_1(x))

        branch3 = self.branch3_2_3(self.branch3_2_3(self.branch3_1(x)))

        branch4 = self.branch4_4_3(self.branch4_3_3(self.branch4_2_3(self.branch4_1(x))))

        out = residual + torch.cat([branch1, branch2, branch3, branch4] , dim=1)
        out = self.relu(out)
        return out

class BiLstmBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BiLstmBlock, self).__init__()
        self.lstm = nn.LSTM(inplanes, planes, num_layers=1, bidirectional=True, batch_first=True)
        self.out_channels = planes * 2

    def forward(self, x):
        out, _ = self.lstm(x)

        return out

class GRUBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(GRUBlock, self).__init__()
        self.gru = nn.GRU(inplanes, planes, num_layers=1, bidirectional=True, batch_first=True)
        self.out_channels = planes * 2

    def forward(self, x):
        out, _ = self.gru(x)

        return out


class entity_avg_feats(nn.Module):
    def __init__(self, feats_in_channel):
        super(entity_avg_feats, self).__init__()
        self.predicate_avgpool = nn.AdaptiveAvgPool1d(1)
        self.feats_num_coef = 2

    def forward(self, subject_feats_in, object_feats_in, bert_feat):
        subject_feats = self.predicate_avgpool(subject_feats_in).permute(0, 2, 1)
        object_feats = self.predicate_avgpool(object_feats_in).permute(0, 2, 1)
        feats_fuse = torch.cat([subject_feats, object_feats], dim=2).squeeze(0)

        return feats_fuse

class entity_max_feats(nn.Module):
    def __init__(self, feats_in_channel):
        super(entity_max_feats, self).__init__()
        self.predicate_maxpool = nn.AdaptiveMaxPool1d(1)
        self.feats_num_coef = 2

    def forward(self, subject_feats_in, object_feats_in, bert_feat):
        subject_feats = self.predicate_maxpool(subject_feats_in).permute(0, 2, 1)
        object_feats = self.predicate_maxpool(object_feats_in).permute(0, 2, 1)
        feats_fuse = torch.cat([subject_feats, object_feats], dim=2).squeeze(0)

        return feats_fuse

class entity_avg_max_feats(nn.Module):
    def __init__(self, feats_in_channel):
        super(entity_avg_max_feats, self).__init__()
        self.predicate_avgpool = nn.AdaptiveAvgPool1d(1)
        self.predicate_maxpool = nn.AdaptiveMaxPool1d(1)
        self.feats_num_coef = 4

    def forward(self, subject_feats_in, object_feats_in, bert_feat):
        subject_avg_feats = self.predicate_avgpool(subject_feats_in).permute(0, 2, 1)
        subject_max_feats = self.predicate_maxpool(subject_feats_in).permute(0, 2, 1)
        
        object_avg_feats = self.predicate_avgpool(object_feats_in).permute(0, 2, 1)
        object_max_feats = self.predicate_maxpool(object_feats_in).permute(0, 2, 1)
        feats_fuse = torch.cat([subject_avg_feats, subject_max_feats, object_avg_feats, object_max_feats], dim=2).squeeze(0)

        return feats_fuse

class entity_avg_max_cls_feats(nn.Module):
    def __init__(self, feats_in_channel):
        super(entity_avg_max_cls_feats, self).__init__()
        self.predicate_avgpool = nn.AdaptiveAvgPool1d(1)
        self.predicate_maxpool = nn.AdaptiveMaxPool1d(1)
        self.feats_num_coef = 5

    def forward(self, subject_feats_in, object_feats_in, bert_feat):
        subject_avg_feats = self.predicate_avgpool(subject_feats_in).permute(0, 2, 1)
        subject_max_feats = self.predicate_maxpool(subject_feats_in).permute(0, 2, 1)
        
        object_avg_feats = self.predicate_avgpool(object_feats_in).permute(0, 2, 1)
        object_max_feats = self.predicate_maxpool(object_feats_in).permute(0, 2, 1)

        cls_feat = bert_feat[:, 0: 1, :]
        feats_fuse = torch.cat([subject_avg_feats, subject_max_feats, object_avg_feats, object_max_feats, cls_feat], dim=2).squeeze(0)

        return feats_fuse

class entity_avg_max_globalmax_feats(nn.Module):
    def __init__(self, feats_in_channel):
        super(entity_avg_max_globalmax_feats, self).__init__()
        self.predicate_avgpool = nn.AdaptiveAvgPool1d(1)
        self.predicate_maxpool = nn.AdaptiveMaxPool1d(1)
        self.feats_num_coef = 5

    def forward(self, subject_feats_in, object_feats_in, bert_feat):
        subject_avg_feats = self.predicate_avgpool(subject_feats_in).permute(0, 2, 1)
        subject_max_feats = self.predicate_maxpool(subject_feats_in).permute(0, 2, 1)
        
        object_avg_feats = self.predicate_avgpool(object_feats_in).permute(0, 2, 1)
        object_max_feats = self.predicate_maxpool(object_feats_in).permute(0, 2, 1)

        cls_feat = self.predicate_maxpool(bert_feat.permute(0, 2, 1)).permute(0, 2, 1)
        feats_fuse = torch.cat([subject_avg_feats, subject_max_feats, object_avg_feats, object_max_feats, cls_feat], dim=2).squeeze(0)

        return feats_fuse

class entity_avg_max_globalavg_feats(nn.Module):
    def __init__(self, feats_in_channel):
        super(entity_avg_max_globalavg_feats, self).__init__()
        self.predicate_avgpool = nn.AdaptiveAvgPool1d(1)
        self.predicate_maxpool = nn.AdaptiveMaxPool1d(1)
        self.feats_num_coef = 5

    def forward(self, subject_feats_in, object_feats_in, bert_feat):
        subject_avg_feats = self.predicate_avgpool(subject_feats_in).permute(0, 2, 1)
        subject_max_feats = self.predicate_maxpool(subject_feats_in).permute(0, 2, 1)
        
        object_avg_feats = self.predicate_avgpool(object_feats_in).permute(0, 2, 1)
        object_max_feats = self.predicate_maxpool(object_feats_in).permute(0, 2, 1)

        cls_feat = self.predicate_avgpool(bert_feat.permute(0, 2, 1)).permute(0, 2, 1)
        feats_fuse = torch.cat([subject_avg_feats, subject_max_feats, object_avg_feats, object_max_feats, cls_feat], dim=2).squeeze(0)

        return feats_fuse

class entity_avg_max_globalavgmax_feats(nn.Module):
    def __init__(self, feats_in_channel):
        super(entity_avg_max_globalavgmax_feats, self).__init__()
        self.predicate_avgpool = nn.AdaptiveAvgPool1d(1)
        self.predicate_maxpool = nn.AdaptiveMaxPool1d(1)
        self.feats_num_coef = 6

    def forward(self, subject_feats_in, object_feats_in, bert_feat):
        subject_avg_feats = self.predicate_avgpool(subject_feats_in).permute(0, 2, 1)
        subject_max_feats = self.predicate_maxpool(subject_feats_in).permute(0, 2, 1)
        
        object_avg_feats = self.predicate_avgpool(object_feats_in).permute(0, 2, 1)
        object_max_feats = self.predicate_maxpool(object_feats_in).permute(0, 2, 1)

        cls_avg_feat = self.predicate_avgpool(bert_feat.permute(0, 2, 1)).permute(0, 2, 1)
        cls_max_feat = self.predicate_maxpool(bert_feat.permute(0, 2, 1)).permute(0, 2, 1)
        feats_fuse = torch.cat([subject_avg_feats, subject_max_feats, object_avg_feats, object_max_feats, cls_max_feat, cls_avg_feat], dim=2).squeeze(0)

        return feats_fuse

class entity_start_end_avg_max_feats(nn.Module):
    def __init__(self, feats_in_channel):
        super(entity_start_end_avg_max_feats, self).__init__()
        self.predicate_avgpool = nn.AdaptiveAvgPool1d(1)
        self.predicate_maxpool = nn.AdaptiveMaxPool1d(1)
        self.feats_num_coef = 8

    def forward(self, subject_feats_in, object_feats_in, bert_feat):
        subject_start_feats = subject_feats_in[:, :, 0: 1].permute(0, 2, 1)
        subject_end_feats = subject_feats_in[:, :, -1: ].permute(0, 2, 1)
        subject_avg_feats = self.predicate_avgpool(subject_feats_in).permute(0, 2, 1)
        subject_max_feats = self.predicate_maxpool(subject_feats_in).permute(0, 2, 1)
        
        object_start_feats = object_feats_in[:, :, 0: 1].permute(0, 2, 1)
        object_end_feats = object_feats_in[:, :, -1: ].permute(0, 2, 1)
        object_avg_feats = self.predicate_avgpool(object_feats_in).permute(0, 2, 1)
        object_max_feats = self.predicate_maxpool(object_feats_in).permute(0, 2, 1)
        feats_fuse = torch.cat([subject_start_feats, subject_end_feats, subject_avg_feats, subject_max_feats, \
            object_start_feats, object_end_feats, object_avg_feats, object_max_feats], dim=2).squeeze(0)

        return feats_fuse

class entity_avg_max_fc_bn_relu_feats(nn.Module):
    def __init__(self, feats_in_channel):
        super(entity_avg_max_fc_bn_relu_feats, self).__init__()
        self.predicate_avgpool = nn.AdaptiveAvgPool1d(1)
        self.predicate_maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(feats_in_channel, feats_in_channel)

        self.feats_num_coef = 4

    def forward(self, subject_feats_in, object_feats_in, bert_feat):
        subject_avg_feats = self.fc(self.predicate_avgpool(subject_feats_in).squeeze(2))
        subject_max_feats = self.fc(self.predicate_maxpool(subject_feats_in).squeeze(2))
        
        object_avg_feats = self.fc(self.predicate_avgpool(object_feats_in).squeeze(2))
        object_max_feats = self.fc(self.predicate_maxpool(object_feats_in).squeeze(2))
        feats_fuse = torch.cat([subject_avg_feats, subject_max_feats, object_avg_feats, object_max_feats], dim=1)

        return feats_fuse

class entity_avg_max_fc_bn_feats(nn.Module):
    def __init__(self, feats_in_channel):
        super(entity_avg_max_fc_bn_feats, self).__init__()
        self.predicate_avgpool = nn.AdaptiveAvgPool1d(1)
        self.predicate_maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(feats_in_channel, feats_in_channel)

        self.feats_num_coef = 4

    def forward(self, subject_feats_in, object_feats_in, bert_feat):
        subject_avg_feats = self.fc(self.predicate_avgpool(subject_feats_in).squeeze(2))
        subject_max_feats = self.fc(self.predicate_maxpool(subject_feats_in).squeeze(2))
        
        object_avg_feats = self.fc(self.predicate_avgpool(object_feats_in).squeeze(2))
        object_max_feats = self.fc(self.predicate_maxpool(object_feats_in).squeeze(2))
        feats_fuse = torch.cat([subject_avg_feats, subject_max_feats, object_avg_feats, object_max_feats], dim=1)

        return feats_fuse

class entity_avg_max_fc_bn_relu_feats_v2(nn.Module):
    def __init__(self, feats_in_channel):
        super(entity_avg_max_fc_bn_relu_feats_v2, self).__init__()
        self.predicate_avgpool = nn.AdaptiveAvgPool1d(1)
        self.predicate_maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(feats_in_channel * 4, feats_in_channel * 4)

        self.feats_num_coef = 4

    def forward(self, subject_feats_in, object_feats_in, bert_feat):
        subject_avg_feats = self.predicate_avgpool(subject_feats_in).permute(0, 2, 1)
        subject_max_feats = self.predicate_maxpool(subject_feats_in).permute(0, 2, 1)
        
        object_avg_feats = self.predicate_avgpool(object_feats_in).permute(0, 2, 1)
        object_max_feats = self.predicate_maxpool(object_feats_in).permute(0, 2, 1)
        feats_fuse = torch.cat([subject_avg_feats, subject_max_feats, object_avg_feats, object_max_feats], dim=2).squeeze(0)
        feats_fuse = self.fc(feats_fuse)
        return feats_fuse

class entity_avg_max_fc_bn_feats_v2(nn.Module):
    def __init__(self, feats_in_channel):
        super(entity_avg_max_fc_bn_feats_v2, self).__init__()
        self.predicate_avgpool = nn.AdaptiveAvgPool1d(1)
        self.predicate_maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(feats_in_channel * 4, feats_in_channel * 4)

        self.feats_num_coef = 4

    def forward(self, subject_feats_in, object_feats_in, bert_feat):
        subject_avg_feats = self.predicate_avgpool(subject_feats_in).permute(0, 2, 1)
        subject_max_feats = self.predicate_maxpool(subject_feats_in).permute(0, 2, 1)
        
        object_avg_feats = self.predicate_avgpool(object_feats_in).permute(0, 2, 1)
        object_max_feats = self.predicate_maxpool(object_feats_in).permute(0, 2, 1)
        feats_fuse = torch.cat([subject_avg_feats, subject_max_feats, object_avg_feats, object_max_feats], dim=2).squeeze(0)
        feats_fuse = self.fc(feats_fuse)
        return feats_fuse


class entity_avg_max_product_feats(nn.Module):
    def __init__(self, feats_in_channel):
        super(entity_avg_max_product_feats, self).__init__()
        self.predicate_avgpool = nn.AdaptiveAvgPool1d(1)
        self.predicate_maxpool = nn.AdaptiveMaxPool1d(1)
        self.feats_num_coef = 6

    def forward(self, subject_feats_in, object_feats_in, bert_feat):
        subject_avg_feats = self.predicate_avgpool(subject_feats_in).permute(0, 2, 1)
        subject_max_feats = self.predicate_maxpool(subject_feats_in).permute(0, 2, 1)
        
        object_avg_feats = self.predicate_avgpool(object_feats_in).permute(0, 2, 1)
        object_max_feats = self.predicate_maxpool(object_feats_in).permute(0, 2, 1)

        subject_feats = torch.cat([subject_avg_feats, subject_max_feats], dim=2)
        object_feats = torch.cat([object_avg_feats, object_max_feats], dim=2)
        product_feats = subject_feats * object_feats
        feats_fuse = torch.cat([subject_feats, object_feats, product_feats], dim=2).squeeze(0)

        return feats_fuse

class fc_bn_relu(nn.Module):
    def __init__(self, feats_in_channel, num_fc):
        super(fc_bn_relu, self).__init__()

        fc_list = []
        self.num_fc = num_fc
        for i in range(num_fc):
            fc_list.extend([nn.Linear(feats_in_channel, feats_in_channel), nn.BatchNorm1d(feats_in_channel), nn.LeakyReLU(0.1)])
        self.fc_list = nn.Sequential(*fc_list)

    def forward(self, x):

        x = self.fc_list(x)
        return x


