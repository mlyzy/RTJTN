import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import argparse

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, bool_bn=False):
        super(TemporalBlock, self).__init__()
        conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        chomp1 = Chomp1d(padding)
        if bool_bn:
            bn1 = nn.BatchNorm1d(n_outputs)
        relu1 = nn.ReLU()
        dropout1 = nn.Dropout(dropout)

        conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        chomp2 = Chomp1d(padding)
        if bool_bn:
            bn2 = nn.BatchNorm1d(n_outputs)
        relu2 = nn.ReLU()
        dropout2 = nn.Dropout(dropout)

        if not bool_bn:
            self.net = nn.Sequential(conv1, chomp1, relu1, dropout1,
                                    conv2, chomp2, relu2, dropout2)
        else:
            self.net = nn.Sequential(conv1, chomp1, bn1, relu1, dropout1,
                                    conv2, chomp2, bn2, relu2, dropout2)            
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalBlock_nochomp(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, bool_bn=False):
        super(TemporalBlock_nochomp, self).__init__()
        conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        if bool_bn:
            bn1 = nn.BatchNorm1d(n_outputs)
        relu1 = nn.ReLU()
        dropout1 = nn.Dropout(dropout)

        conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        if bool_bn:
            bn2 = nn.BatchNorm1d(n_outputs)
        relu2 = nn.ReLU()
        dropout2 = nn.Dropout(dropout)

        if not bool_bn:
            self.net = nn.Sequential(conv1, relu1, dropout1,
                                    conv2, relu2, dropout2)
        else:
            self.net = nn.Sequential(conv1, bn1, relu1, dropout1,
                                    conv2, bn2, relu2, dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        self.out_channels = num_channels[-1]
    def forward(self, x):
        return self.network(x)

class TemporalConvNet_nochomp(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet_nochomp, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock_nochomp(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-2) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        self.out_channels = num_channels[-1]

    def forward(self, x):
        return self.network(x)

class TemporalConvNet_BN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet_BN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, bool_bn=True)]

        self.network = nn.Sequential(*layers)

        self.out_channels = num_channels[-1]
    def forward(self, x):
        return self.network(x)

class TemporalConvNet_nochomp_BN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet_nochomp_BN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock_nochomp(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-2) * dilation_size, dropout=dropout, bool_bn=True)]

        self.network = nn.Sequential(*layers)

        self.out_channels = num_channels[-1]
    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):

    def __init__(self, input_size, output_size, num_channels,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1, tied_weights=False):
        super(TCN, self).__init__()
        self.encoder = nn.Embedding(output_size, input_size)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], output_size)
        if tied_weights:
            if num_channels[-1] != input_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb = self.drop(self.encoder(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='TemporalConvNet', type=str, help="model.")
    args = parser.parse_args()

    model = eval(args.model)(768, [256, 256, 256])

    inputs = torch.randn(4, 768, 128)
    out = model(inputs)
    # torch.save(model.state_dict(), './model.pth')
    print(out.shape)
