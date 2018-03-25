#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/9/17
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : train_test_classifiers.py

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
import datetime


def train_test_SVM(args):
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
    test_loader = data_utils.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model.eval()
    features_train = []

    start_deep = datetime.datetime.now()

    for sample_batched in train_loader:
        if args.use_gpu and torch.cuda.is_available():
            x_1 = torch.autograd.Variable(sample_batched['glove'].cuda())
            x_2 = torch.autograd.Variable(sample_batched['res'].cuda())
            x_3 = torch.autograd.Variable(sample_batched['bitrate'].cuda())
            x_4 = torch.autograd.Variable(sample_batched['gender'].cuda())
            x_5 = torch.autograd.Variable(sample_batched['age'].cuda())

        else:
            x_1 = torch.autograd.Variable(sample_batched['glove'])
            x_2 = torch.autograd.Variable(sample_batched['res'])
            x_3 = torch.autograd.Variable(sample_batched['bitrate'])
            x_4 = torch.autograd.Variable(sample_batched['gender'])
            x_5 = torch.autograd.Variable(sample_batched['age'])
        _, fc2_train = model(x_1, x_2, x_3, x_4, x_5)
        features_train.append(fc2_train.data.cpu().numpy())
    train_features = np.concatenate(features_train, 0)

    total_deep = float((datetime.datetime.now() - start_deep).total_seconds()) / float(len(train_data))
    print("DeepQoE total cost {}s".format(total_deep))
    print(len(train_data))

    clf = cfg.CLASSIFIER[args.classifier]
    clf.fit(train_features, y_train)

    features_test = []

    start_deep = datetime.datetime.now()

    for sample_batched in test_loader:
        if args.use_gpu and torch.cuda.is_available():
            x_1 = torch.autograd.Variable(sample_batched['glove'].cuda())
            x_2 = torch.autograd.Variable(sample_batched['res'].cuda())
            x_3 = torch.autograd.Variable(sample_batched['bitrate'].cuda())
            x_4 = torch.autograd.Variable(sample_batched['gender'].cuda())
            x_5 = torch.autograd.Variable(sample_batched['age'].cuda())

        else:
            x_1 = torch.autograd.Variable(sample_batched['glove'])
            x_2 = torch.autograd.Variable(sample_batched['res'])
            x_3 = torch.autograd.Variable(sample_batched['bitrate'])
            x_4 = torch.autograd.Variable(sample_batched['gender'])
            x_5 = torch.autograd.Variable(sample_batched['age'])
        _, fc2_test = model(x_1, x_2, x_3, x_4, x_5)
        features_test.append(fc2_test.data.cpu().numpy())
    test_features = np.concatenate(features_test, 0)

    total_deep = float((datetime.datetime.now() - start_deep).total_seconds()) / float(len(test_data))
    print("DeepQoE total cost {}s".format(total_deep))
    print(len(test_data))

    prediction = clf.predict(test_features)
    acc = accuracy_score(prediction, y_test)
    print ("{} uses DeepQoE features can get {}%".format(cfg.CLASSIFIER_NAME[args.classifier], acc * 100.0))

    clf_ori = cfg.CLASSIFIER[args.classifier]
    clf_ori.fit(x_train.astype(float), y_train)
    prediction_ori = clf_ori.predict(x_test.astype(float))
    acc_ori = accuracy_score(prediction_ori, y_test)
    print("{} uses original features can get {}%".format(cfg.CLASSIFIER_NAME[args.classifier], acc_ori * 100.0))

if __name__ == '__main__':
    train_test_SVM(parse_arguments(sys.argv[1:]))
