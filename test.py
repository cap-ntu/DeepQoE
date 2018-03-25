#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 28/8/17
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : test.py

from __future__ import print_function

import torch
import pickle
import sys
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import torch.utils.data as data_utils
from DeepQoE.config import cfg, parse_arguments
from DeepQoE.nets import *
from DeepQoE.data_loader import QoETextDataset


def test_confusion_matrix(args):
    model = HybridNN()
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_TEXT))

    with open(cfg.EMBEDDING_DATA, 'rb') as f:
        data = pickle.load(f)
        x, y = data[0], data[1]

    train_size = int(cfg.TRAIN_RATIO * len(x))

    x_train = x[:train_size]
    y_train = y[:train_size]
    x_test = x[train_size:]
    y_test = y[train_size:]
    print (y_test)

    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        model.cuda()

    train_data = QoETextDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

    test_data = QoETextDataset(x_test, y_test)
    test_loader = data_utils.DataLoader(test_data, batch_size=10, shuffle=False)

    model.eval()

    for sample_batched in test_loader:
        if args.use_gpu and torch.cuda.is_available():
            x_1 = torch.autograd.Variable(sample_batched['glove'].cuda())
            x_2 = torch.autograd.Variable(sample_batched['res'].cuda())
            x_3 = torch.autograd.Variable(sample_batched['bitrate'].cuda())
            x_4 = torch.autograd.Variable(sample_batched['gender'].cuda())
            x_5 = torch.autograd.Variable(sample_batched['age'].cuda())
            target = torch.autograd.Variable(sample_batched['label'].cuda())
        else:
            x_1 = torch.autograd.Variable(sample_batched['glove'])
            x_2 = torch.autograd.Variable(sample_batched['res'])
            x_3 = torch.autograd.Variable(sample_batched['bitrate'])
            x_4 = torch.autograd.Variable(sample_batched['gender'])
            x_5 = torch.autograd.Variable(sample_batched['age'])
            target = torch.autograd.Variable(sample_batched['label'])

        output, _ = model(x_1, x_2, x_3, x_4, x_5)
        pred = output.data.max(1, keepdim=True)[1]
        print ("True: {}".format(target))
        print ("Predict: {}".format(pred))

if __name__ == '__main__':
    test_confusion_matrix(parse_arguments(sys.argv[1:]))