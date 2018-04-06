#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 14/12/17
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : train_test_MOS.py

from __future__ import print_function

import os
import sys
import numpy as np
import pickle
import logging
import torch.optim as optim
import torch.utils.data as data_utils
from DeepQoE.nets import *
from DeepQoE.config import cfg, parse_arguments
from DeepQoE.data_loader import QoEMOSDataset


def main(args):
    model = HybridRR()
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        model.cuda()
    print(model)

    model.train()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-08, weight_decay=1e-6)
    with open(cfg.MOS_DATA, 'rb') as f:
        data = pickle.load(f)
        x = data[:, 0:data.shape[1] - 1]
        y = np.array(data[:, -1], np.float)

    # x_train = np.concatenate((x[0:36], x[54:]), axis=0)
    # y_train = np.concatenate((y[0:36], y[54:]), axis=0)
    x_train = x[0:54]
    y_train = y[0:54]

    x_test = x[54:]
    y_test = y[54:]

    train_data = QoEMOSDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(train_data, batch_size=54, shuffle=False)
    for epoch in range(args.epochs):
        for batch_idx, sample_batched in enumerate(train_loader):
            pid = os.getpid()
            if args.use_gpu and torch.cuda.is_available():
                x_1 = torch.autograd.Variable(sample_batched['glove'].cuda())
                x_2 = torch.autograd.Variable(sample_batched['res'].cuda())
                x_3 = torch.autograd.Variable(sample_batched['bitrate'].cuda())
                target = torch.autograd.Variable(sample_batched['label'].cuda())
            else:
                x_1 = torch.autograd.Variable(sample_batched['glove'])
                x_2 = torch.autograd.Variable(sample_batched['res'])
                x_3 = torch.autograd.Variable(sample_batched['bitrate'])
                target = torch.autograd.Variable(sample_batched['label'])
            # print (target)
            optimizer.zero_grad()
            prediction, _ = model(x_1, x_2, x_3)
            loss = F.mse_loss(prediction, target.float())

            loss.backward()
            optimizer.step()

            if batch_idx % 3 == 0:
                print('{}\tTrain Epoch: {} \tLoss: {:.6f}'.format(
                    pid, epoch, loss.data[0]))
    torch.save(model.state_dict(), cfg.MODEL_SAVE_MOS)

    # test processing
    test_data = QoEMOSDataset(x_test, y_test)
    test_loader = data_utils.DataLoader(test_data, batch_size=18, shuffle=False)
    model.eval()
    test_loss = 0
    for sample_batched in test_loader:
        if args.use_gpu and torch.cuda.is_available():
            x_1 = torch.autograd.Variable(sample_batched['glove'].cuda())
            x_2 = torch.autograd.Variable(sample_batched['res'].cuda())
            x_3 = torch.autograd.Variable(sample_batched['bitrate'].cuda())
            target = torch.autograd.Variable(sample_batched['label'].cuda())

        else:
            x_1 = torch.autograd.Variable(sample_batched['glove'])
            x_2 = torch.autograd.Variable(sample_batched['res'])
            x_3 = torch.autograd.Variable(sample_batched['bitrate'])
            target = torch.autograd.Variable(sample_batched['label'])

        output, _ = model(x_1, x_2, x_3)
        test_loss += F.mse_loss(output, target.float(), size_average=False).data[0]
        # print (output)
        print (target.float())

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
