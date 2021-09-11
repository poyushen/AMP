#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class SEblock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels//reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels//reduction, channels)
        self.sigmoid = nn.Sigmoid()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)

        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        input_x = x
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(x.size(0), x.size(1), 1)

        return input_x * x


class BasicLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=(kernel_size//2))
        self.relu = nn.ReLU(inplace=True)

        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return x


class SELayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, reduction=4):
        super(SELayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=(kernel_size//2))
        self.relu = nn.ReLU(inplace=True)
        self.se = SEblock(channels=out_channels, reduction=reduction)

        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.se(x)

        return x


class SENet(nn.Module):
    def __init__(self, pool=True):
        super(SENet, self).__init__()
        self.channels = [128, 128, 64, 64]
        self.dropout = 0.5
        self.input_dim = 21
        self.kernel_sizes = [17, 9, 5, 5]
        self.reduction = 4
        self.pool = pool

        self.layer1 = SELayer(in_channels=self.input_dim, out_channels=self.channels[0], kernel_size=self.kernel_sizes[0])
        self.layer2 = SELayer(in_channels=self.channels[0], out_channels=self.channels[1], kernel_size=self.kernel_sizes[1])
        self.layer3 = SELayer(in_channels=self.channels[1], out_channels=self.channels[2], kernel_size=self.kernel_sizes[2])
        self.layer4 = SELayer(in_channels=self.channels[2], out_channels=self.channels[3], kernel_size=self.kernel_sizes[3])

        self.avgpool = nn.AvgPool1d(kernel_size=5, stride=5)
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(self.dropout)

        self.fc = nn.Linear(self.channels[-1], 2)

        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1) # n*21*L

        x1 = self.layer1(x) # n*128*L
        if self.pool:
            x1 = self.avgpool(x1) # n*128*(L/5)

        x2 = self.drop(x1) # n*128*(L/5)
        x2 = self.layer2(x2) # n*128*(L/5)

        x3 = self.drop(x2) #n*128*(L/5)
        x3 = self.layer3(x3) # n*64*(L/5)

        x4 = self.drop(x3) # n*64*(L/5)
        x4 = self.layer4(x4) # n*64*(L/5)

        out = self.globalavgpool(x4) # n*64*1
        out = out.view(out.size(0), -1) # n*64

        out = self.drop(out) # n*128
        out = self.fc(out) # n*2

        return out.squeeze()


class ConcatNet(nn.Module):
    def __init__(self):
        super(ConcatNet, self).__init__()
        self.channels = [128, 128, 64, 64]
        self.dropout = 0.5
        self.input_dim = 21
        self.kernel_sizes = [17, 9, 5, 5]
        self.node = 32
        self.reduction = 4

        self.denselayer1 = SELayer(in_channels=self.input_dim, out_channels=self.channels[0], kernel_size=self.kernel_sizes[0], reduction=self.reduction)
        # self.denselayer1 = BasicLayer(in_channels=self.input_dim, out_channels=self.channels[0], kernel_size=self.kernel_sizes[0])

        self.denselayer2 = BasicLayer(in_channels=sum(self.channels[:1]), out_channels=self.channels[1], kernel_size=self.kernel_sizes[1])
        self.denselayer3 = BasicLayer(in_channels=sum(self.channels[:2]), out_channels=self.channels[2], kernel_size=self.kernel_sizes[2])
        self.denselayer4 = BasicLayer(in_channels=sum(self.channels[1:3]), out_channels=self.channels[3], kernel_size=self.kernel_sizes[3])


        self.reslayer1 = SELayer(in_channels=self.input_dim, out_channels=self.channels[0], kernel_size=self.kernel_sizes[0], reduction=self.reduction)
        # self.reslayer1 = BasicLayer(in_channels=self.input_dim, out_channels=self.channels[0], kernel_size=self.kernel_sizes[0])

        self.reslayer2 = BasicLayer(in_channels=self.channels[0], out_channels=self.channels[1], kernel_size=self.kernel_sizes[1])
        self.reslayer3 = BasicLayer(in_channels=self.channels[1], out_channels=self.channels[2], kernel_size=self.kernel_sizes[2])
        self.reslayer4 = BasicLayer(in_channels=self.channels[2], out_channels=self.channels[3], kernel_size=self.kernel_sizes[3])


        self.avgpool = nn.AvgPool1d(kernel_size=5, stride=5)
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(self.dropout)

        self.fc1 = nn.Linear(self.channels[3]*2, self.node)
        self.fc2 = nn.Linear(self.node, 2)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)

        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1) # n*21*L

        dense_x1 = self.denselayer1(x) # n*128*L
        dense_x1 = self.avgpool(dense_x1) # n*128*(L/5)

        dense_x2 = self.drop(dense_x1) # n*128*(L/5)
        dense_x2 = self.denselayer2(dense_x2) # n*128*(L/5)

        dense_x3 = torch.cat((dense_x1, dense_x2), 1) # n*256*(L/5)
        dense_x3 = self.drop(dense_x3) # n*256*(L/5)
        dense_x3 = self.denselayer3(dense_x3) # n*64*(L/5)

        dense_x4 = torch.cat((dense_x2, dense_x3), 1) # n*384*(L/5)
        dense_x4 = self.drop(dense_x4) # n*384*(L/5)
        dense_x4 = self.denselayer4(dense_x4) # n*64*(L/5)

        dense_out = self.globalavgpool(dense_x4) # n*64*1
        dense_out = dense_out.view(dense_out.size(0), -1) # n*64


        res_x1 = self.reslayer1(x) # n*128*L
        res_x1 = self.avgpool(res_x1) # n*128*(L/5)

        res_x2 = self.drop(res_x1) # n*128*(L/5)
        res_x2 = self.reslayer2(res_x2) # n*128*(L/5)

        res_x3 = torch.add(res_x1, res_x2) # n*128*(L/5)
        res_x3 = self.drop(res_x3) #n*128*(L/5)
        res_x3 = self.reslayer3(res_x3) # n*64*(L/5)

        res_x4 = self.drop(res_x3) # n*64*(L/5)
        res_x4 = self.reslayer4(res_x4) # n*64*(L/5)

        res_out = self.globalavgpool(res_x4) # n*64*1
        res_out = res_out.view(res_out.size(0), -1) # n*64

        out = torch.cat((dense_out, res_out), 1) # n*128
        out = self.drop(out) # n*128
        out = self.fc1(out) # n*32
        out = self.fc2(out) # n*2

        return out.squeeze()


class ScannerNet(nn.Module):
    def __init__(self):
        super(ScannerNet, self).__init__()

        self.embedding = nn.Embedding(21, 128)
        self.conv = nn.Conv1d(in_channels=128, out_channels=64,
                               kernel_size=16, stride=1, padding=(16//2))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=5, stride=5)
        self.lstm = nn.LSTM(input_size=64, hidden_size=100, batch_first=True, bias=True, dropout=0.1)
        self.linear = nn.Linear(100, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1) ## n*128*200
        x = self.conv(x) ## n*64*201
        x = self.relu(x)
        x = self.pool(x) ## n*64*40
        x = x.transpose(2, 1) ## n*40*64
        x, (_, _) = self.lstm(x) ## n*40*100
        x = x[:, -1, :] ## n*100

        x = self.linear(x)

        return x.squeeze()
