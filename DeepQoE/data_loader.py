#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 7/9/17
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : data_loader.py

import torch
import numpy as np
from torch.utils.data import Dataset
import pickle


class QoETextDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        glove_index = torch.from_numpy(self.x[idx, 0:50].astype(np.float)).float()
        res = self.x[idx, [-4]].astype(np.int)
        bitrate = torch.from_numpy(self.x[idx, [-3]].astype(np.float)).float()
        gender = self.x[idx, [-2]].astype(np.int)
        age = torch.from_numpy(self.x[idx, [-1]].astype(np.float)).float()

        label = self.y[idx]
        sample = {'glove': glove_index, 'res': res, 'bitrate': bitrate,
                  'gender': gender, 'age': age, 'label': label}

        return sample


class QoENFLXDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        VQA = torch.from_numpy(self.x[idx, [0]].astype(np.float)).float()
        R1 = torch.from_numpy(self.x[idx, [1]].astype(np.float)).float()
        R2 = self.x[idx, [2]].astype(np.int)
        M = torch.from_numpy(self.x[idx, [3]].astype(np.float)).float()
        I = torch.from_numpy(self.x[idx, [4]].astype(np.float)).float()
        label = self.y[idx]
        sample = {'VQA': VQA, 'R1': R1, 'R2': R2, 'Mem': M, 'Impair': I, 'label': label}

        return sample





class QoEMOSDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        glove_index = torch.from_numpy(self.x[idx, 0:50].astype(np.float)).float()
        res = self.x[idx, [-2]].astype(np.int)
        bitrate = torch.from_numpy(self.x[idx, [-1]].astype(np.float)).float()
        label = self.y[idx]
        sample = {'glove': glove_index, 'res': res, 'bitrate': bitrate, 'label': label}

        return sample

class QoEVideoDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        video = torch.from_numpy(self.x[idx, 0:-1].astype(np.float)).float()
        res = self.x[idx, [-1]].astype(np.int)
        label = self.y[idx]
        sample = {'video': video, 'res': res, 'label': label}

        return sample

