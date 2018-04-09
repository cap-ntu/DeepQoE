#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/12/17
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : train_test_video.py

from __future__ import print_function

import os
import sys
import numpy as np
import pickle
from sklearn import preprocessing
import torch.optim as optim
import torch.utils.data as data_utils
from DeepQoE.nets import *
from DeepQoE.config import cfg, parse_arguments
from DeepQoE.data_loader import QoEVideoDataset

def shuffle_data(x, y):
    sh = np.arange(x.shape[0])
    np.random.shuffle(sh)
    x = x[sh]
    y = y[sh]
    return x, y

def main(args):

    if args.model_number == 0:
        model = HybridNN()
    elif args.model_number == 1:
        model = C3DHybridNN()
    else:
        model = HybridRR()
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        model.cuda()
    print(model)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)

    with open(cfg.JND_DATA, 'rb') as f:
        data = pickle.load(f)

    train_test = np.array(data)
    x_temp = train_test[:, 0:train_test.shape[1] - 1]
    y_temp = np.array(train_test[:, train_test.shape[1] - 1], np.int)
    le = preprocessing.LabelEncoder()
    le.fit(y_temp)
    encode_lable = le.transform(y_temp)

    print('Encoder {}'.format('lable') + '\n~~~~~~~~~~~~~~~~~~~~~~~')
    print(x_temp, y_temp)

    x, y = shuffle_data(x_temp, encode_lable)

    train_size = int(cfg.TRAIN_RATIO * len(x))

    x_train = x[:train_size]
    x_test = x[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    train_data = QoEVideoDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    for epoch in range(args.epochs):
        for batch_idx, sample_batched in enumerate(train_loader):
            pid = os.getpid()
            if args.use_gpu and torch.cuda.is_available():
                x_1 = torch.autograd.Variable(sample_batched['video'].cuda())
                x_2 = torch.autograd.Variable(sample_batched['res'].cuda())
                target = torch.autograd.Variable(sample_batched['label'].cuda())
            else:
                x_1 = torch.autograd.Variable(sample_batched['video'])
                x_2 = torch.autograd.Variable(sample_batched['res'])
                target = torch.autograd.Variable(sample_batched['label'])
            # print (target)
            optimizer.zero_grad()
            prediction, _ = model(x_1, x_2)
            # print(prediction)
            if args.model_number == 2:
                loss = F.mse_loss(prediction, target)
            else:
                loss = F.nll_loss(prediction, target)

            loss.backward()
            optimizer.step()
            # print(model.layer1_glove.weight)

            if batch_idx % 3 == 0:
                print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    pid, epoch, batch_idx * len(data), len(train_loader.dataset),
                                100. * batch_idx / len(train_loader), loss.data[0]))
    torch.save(model.state_dict(), cfg.MODEL_SAVE_VIDEO)

    # test processing
    test_data = QoEVideoDataset(x_test, y_test)
    test_loader = data_utils.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    model.eval()
    test_loss = 0
    correct = 0
    for sample_batched in test_loader:
        if args.use_gpu and torch.cuda.is_available():
            x_1 = torch.autograd.Variable(sample_batched['video'].cuda())
            x_2 = torch.autograd.Variable(sample_batched['res'].cuda())
            target = torch.autograd.Variable(sample_batched['label'].cuda())
        else:
            x_1 = torch.autograd.Variable(sample_batched['video'])
            x_2 = torch.autograd.Variable(sample_batched['res'])
            target = torch.autograd.Variable(sample_batched['label'])
        output, _ = model(x_1, x_2)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))