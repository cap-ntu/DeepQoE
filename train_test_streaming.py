#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/3/18 3:40 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : train_test_streaming.py

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
from scripts.generate_pretrained_models import generate_streaming_data
from DeepQoE.data_loader import QoENFLXDataset
from scipy.stats import spearmanr as sr
import matplotlib.pyplot as plt


def main(args):
    model = HybridStreaming()
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        model.cuda()
    print(model)

    optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-08, weight_decay=1e-6)
    x_train, y_train, x_test, y_test, feature_labels = generate_streaming_data()
    # print(x_train)

    train_data = QoENFLXDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(train_data, batch_size=64, shuffle=False)

    test_data = QoENFLXDataset(x_test, y_test)
    test_loader = data_utils.DataLoader(test_data, batch_size=64, shuffle=False)

    for epoch in range(args.epochs):
        train(train_loader, model, optimizer, epoch)

    output = test(test_loader, model)
    predict = output.cpu().data.numpy()
    show_results(x_test, x_test, y_test, 'DeepQoE', feature_labels, predict)

    # torch.save(model.state_dict(), cfg.MODEL_SAVE_MOS)


def train(train_loader, model, optimizer, epoch):
    model.train()
    for batch_idx, sample_batched in enumerate(train_loader):
        pid = os.getpid()

        x_1 = torch.autograd.Variable(sample_batched['VQA'].cuda())
        x_2 = torch.autograd.Variable(sample_batched['R1'].cuda())
        x_3 = torch.autograd.Variable(sample_batched['R2'].cuda())
        x_4 = torch.autograd.Variable(sample_batched['Mem'].cuda())
        x_5 = torch.autograd.Variable(sample_batched['Impair'].cuda())
        target = torch.autograd.Variable(sample_batched['label'].cuda())

        optimizer.zero_grad()
        prediction, _ = model(x_1, x_2, x_3, x_4, x_5)
        loss = F.mse_loss(prediction, target.float())

        loss.backward()
        optimizer.step()

        if batch_idx % 3 == 0:
            print('{}\tTrain Epoch: {} \tLoss: {:.6f}'.format(
                pid, epoch, loss.data[0]))


def test(test_loader, model):
    model.eval()
    test_loss = 0
    for sample_batched in test_loader:
        x_1 = torch.autograd.Variable(sample_batched['VQA'].cuda())
        x_2 = torch.autograd.Variable(sample_batched['R1'].cuda())
        x_3 = torch.autograd.Variable(sample_batched['R2'].cuda())
        x_4 = torch.autograd.Variable(sample_batched['Mem'].cuda())
        x_5 = torch.autograd.Variable(sample_batched['Impair'].cuda())
        target = torch.autograd.Variable(sample_batched['label'].cuda())

        output, _ = model(x_1, x_2, x_3, x_4, x_5)
        test_loss += F.mse_loss(output, target.float(), size_average=False).data[0]
        # print (output)
        print(target.float())

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    return output


def show_results(X_test, X_test_before_scaling, y_test, regressor_name, feature_labels, answer):

    if cfg.QUALITY_MODEL + "_" + cfg.POOLING_TYPE in feature_labels:
        position_vqa = feature_labels.index(cfg.QUALITY_MODEL + "_" + cfg.POOLING_TYPE)

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    plt.title("before: " + format(sr(y_test, X_test[:, position_vqa].reshape(-1, 1))[0], '.4f'))
    plt.scatter(y_test, X_test_before_scaling[:, position_vqa].reshape(-1, 1))
    plt.grid()
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    ax1.set_aspect((x1 - x0) / (y1 - y0))
    plt.ylabel("predicted QoE")
    plt.xlabel("MOS")
    plt.show()

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    plt.title("after: " + format(sr(y_test, answer.reshape(-1, 1))[0], '.4f'))
    plt.scatter(y_test, answer.reshape(-1, 1))
    plt.grid()
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    ax1.set_aspect((x1 - x0) / (y1 - y0))
    plt.ylabel("predicted QoE")
    plt.xlabel("MOS")
    plt.show()

    print("SROCC before (" + str(cfg.QUALITY_MODEL) + "): " + str(sr(y_test, X_test[:, position_vqa].reshape(-1, 1))[0]))
    print("SROCC using DeepQoE (" + str(cfg.QUALITY_MODEL) + " + " + regressor_name + "): " + str(
        sr(y_test, answer.reshape(-1, 1))[0]))


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
