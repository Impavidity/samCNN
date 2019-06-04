import torch
from torch import nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_of_conv, in_channels, out_channels, kernel_size, in_features, out_features=None, stride=1,
                 dilation=1, groups=1, bias=True, active_func=F.relu, pooling=F.max_pool1d,
                 dropout=0.5, padding_strategy="default", padding_list=None, fc_layer=True,
                 include_map=False, k_max_pooling=False, k=1):
        """

        :param num_of_conv: Follow kim cnn idea
        :param kernel_size: if is int type, then make it into list, length equals to num_of_conv
                     if list type, then check the length of it, should has length of num_of_conv
        :param out_features: feature size
        """
        super(SimpleCNN, self).__init__()
        if type(kernel_size) == int:
            kernel_size = [kernel_size]
        if len(kernel_size) != num_of_conv:
            print("Number of kernel_size should be same num_of_conv")
            exit(1)
        if padding_list == None:
            if padding_strategy == "default":
                padding_list = [(k_size - 1, 0) for k_size in kernel_size]
        self.include_map = include_map

        self.conv = nn.ModuleList([nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=(k_size, in_features),
                                             stride=stride,
                                             padding=padding,
                                             dilation=dilation,
                                             groups=groups,
                                             bias=bias)
                                   for k_size, padding in zip(kernel_size, padding_list)])
        self.pooling = pooling
        self.k_max_pooling = k_max_pooling
        self.k = k if k_max_pooling else 1
        self.active_func = active_func
        self.fc_layer = fc_layer
        if fc_layer:
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(num_of_conv * out_channels * self.k, out_features)

    def kmax_pooling(self, x, dim, k):
        index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
        return x.gather(dim, index)

    def forward(self, input):
        batch_size = input.size(0)
        if len(input.size()) == 3:
            input = input.unsqueeze(1)
        # input = (batch, in_channels, sent_len, word_dim)
        x_map = [self.active_func(conv(input)).squeeze(3) for conv in self.conv]
        # (batch, channel_output, ~=sent_len) * Ks
        if self.k_max_pooling:
            x = [self.kmax_pooling(i, 2, self.k).view(batch_size, -1) for i in x_map]
        else:
            x = [self.pooling(i, i.size(2)).squeeze(2) for i in x_map]  # max-over-time pooling
        x = torch.cat(x, 1)  # (batch, out_channels * Ks)
        if self.fc_layer:
            x = self.dropout(x)
            x = self.fc(x)
        if self.include_map == False:
            return x
        else:
            return x, x_map


class MLP(nn.Linear):
    def __init__(self, in_features, out_features, activation=None, dropout=0.0, bias=True):
        super(MLP, self).__init__(in_features, out_features, bias)
        if activation is None:
            self._activate = None
        else:
            if not callable(activation):
                raise ValueError("activation must be callable, but got {}".format(type(activation)))
            self._activate = activation
        assert dropout == 0 or type(dropout) == float
        self._dropout_ratio = dropout
        if dropout > 0:
            self._dropout = nn.Dropout(p=self._dropout_ratio)

    def forward(self, x):
        size = x.size()
        if len(size) > 2:
            y = super(MLP, self).forward(
                x.contiguous().view(-1, size[-1]))
            y = y.view(size[0:-1] + (-1,))
        else:
            y = super(MLP, self).forward(x)
        if self._activate is not None:
            y = self._activate(y)
        if self._dropout_ratio > 0:
            return self._dropout(y)
        else:
            return y


class SequenceCNN(nn.Module):
    def __init__(self, num_of_conv, in_channels, out_channels, kernel_size, in_features, stride=1,
                 dilation=1, groups=1, bias=True):
        super(SequenceCNN, self).__init__()
        if type(kernel_size) == int:
            kernel_size = [kernel_size]
        if len(kernel_size) != num_of_conv:
            print("Number of kernel_size should be same num_of_conv")
            exit(1)
        for k_size in kernel_size:
            if k_size % 2 == 0:
                print("The kernel size is better to be odd")
                exit(1)
        padding_list = [(int(k_size / 2), 0) for k_size in kernel_size]
        self.conv = nn.ModuleList([nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=(k_size, in_features),
                                             stride=stride,
                                             padding=padding,
                                             dilation=dilation,
                                             groups=groups,
                                             bias=bias)
                                   for k_size, padding in zip(kernel_size, padding_list)])

    def forward(self, input):
        if len(input.size()) == 3:
            input = input.unsqueeze(1)
        # input = (batch, in_channels, sent_len, word_dim)
        x = [conv(input).squeeze(3).transpose(1, 2) for conv in self.conv]
        # x = (batch, sent_len, out_channels) * Ks
        x = torch.cat(x, dim=2)
        return x