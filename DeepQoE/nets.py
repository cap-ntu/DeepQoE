#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 29/8/17
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : nets.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from DeepQoE.config import cfg

class HybridNN(nn.Module):
    def __init__(self):
        super(HybridNN, self).__init__()
        self.layer1_glove = nn.Linear(50, 5)
        self.layer1_res = nn.Embedding(3, 8)
        self.layer1_bit = nn.Linear(1, cfg.DIMENSION)
        self.layer1_gender = nn.Embedding(2, 1)
        self.layer1_age = nn.Linear(1, 1)

        # self.bn_0 = nn.BatchNorm2d(16)
        # self.conv1 = nn.Conv1d(1, 1, 1, stride=1)

        self.fc1 = nn.Linear(15+cfg.DIMENSION, 512)

        # self.bn_1 = nn.BatchNorm2d(512)

        self.fc2 = nn.Linear(512, 256)
        # self.bn_2 = nn.BatchNorm2d(256)

        self.fc3 = nn.Linear(256, 5)

    def forward(self, x1, x2, x3, x4, x5):
        x_res = self.layer1_res(x2).view(-1, 8)
        x_gender = self.layer1_gender(x4).view(-1, 1)
        h = torch.cat((self.layer1_glove(x1), x_res,
                       self.layer1_bit(x3), x_gender,
                       self.layer1_age(x5)), 1)

        # h = torch.stack([h], dim=1)
        # h = self.conv1(h)
        # h = torch.squeeze(h)
        # h = F.tanh(h)

        h = F.tanh(self.fc1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        fc2 = F.tanh(self.fc2(h))
        h = F.dropout(fc2, p=0.5, training=self.training)
        h = F.log_softmax(self.fc3(h))
        return h, fc2


class C3DHybridNN(nn.Module):
    def __init__(self):
        super(C3DHybridNN, self).__init__()
        self.layer1_content = nn.Linear(4096, 4088)
        self.layer1_res = nn.Embedding(4, 8)

        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 4)

    def forward(self, x1, x2):
        x_res = self.layer1_res(x2).view(-1, 8)
        h = torch.cat((self.layer1_content(x1), x_res), 1)
        h = F.tanh(h)
        h = F.tanh(self.fc1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        h = F.tanh(self.fc2(h))
        h = F.dropout(h, p=0.5, training=self.training)
        fc3 = F.tanh(self.fc3(h))
        h = F.dropout(fc3, p=0.5, training=self.training)
        h = F.log_softmax(self.fc4(h))
        return h, fc3


class HybridRR(nn.Module):
    def __init__(self):
        super(HybridRR, self).__init__()
        self.layer1_glove = nn.Linear(50, 5)
        self.layer1_res = nn.Embedding(3, 8)
        self.layer1_bit = nn.Linear(1, 1)

        self.fc1 = nn.Linear(14, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x1, x2, x3):
        x_res = self.layer1_res(x2).view(-1, 8)
        h = torch.cat((self.layer1_glove(x1), x_res,
                       self.layer1_bit(x3)), 1)
        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        fc2 = F.relu(self.fc2(h))
        h = F.dropout(fc2, p=0.5, training=self.training)
        h = self.fc3(h)
        return h, fc2


class HybridStreaming(nn.Module):
    def __init__(self):
        super(HybridStreaming, self).__init__()
        self.layer1_VQA = nn.Linear(1, 20)
        self.layer1_R1 = nn.Linear(1, 5)
        self.layer1_R2 = nn.Embedding(3, 5)
        self.layer1_M = nn.Linear(1, 10)
        self.layer1_I = nn.Linear(1, 10)

        self.fc1 = nn.Linear(50, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x1, x2, x3, x4, x5):
        x_R2 = self.layer1_R2(x3).view(-1, 5)
        h = torch.cat((self.layer1_VQA(x1), self.layer1_R1(x2), x_R2,
                       self.layer1_M(x4), self.layer1_I(x5)), 1)
        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        fc2 = F.relu(self.fc2(h))
        h = F.dropout(fc2, p=0.5, training=self.training)
        h = self.fc3(h)
        return h, fc2




