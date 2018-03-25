#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 28/8/17
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : train_test.py

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
from DeepQoE.data_loader import QoETextDataset


def main(args):
    log_file = 'log/' + cfg.LOG + '_' + str(cfg.DIMENSION) + '.txt'
    print(log_file)
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w')

    result = 0.0
    for i in range(10):
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
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        with open(cfg.EMBEDDING_DATA, 'rb') as f:
            data = pickle.load(f)
            x, y = data[0], data[1]

        train_size = int(cfg.TRAIN_RATIO * len(x))

        x_train = x[:train_size]
        y_train = y[:train_size]

        x_test = x[train_size:]
        y_test = y[train_size:]




        train_data = QoETextDataset(x_train, y_train)
        train_loader = data_utils.DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
        # print (train_loader)
        # for batch_idx, sample_batched in enumerate(train_loader):
        #     print (sample_batched)
        for epoch in range(args.epochs):
            for batch_idx, sample_batched in enumerate(train_loader):
                pid = os.getpid()
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
                # print (target)
                optimizer.zero_grad()
                prediction, _ = model(x_1, x_2, x_3, x_4, x_5)
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
        torch.save(model.state_dict(), cfg.MODEL_SAVE_TEXT)

        # test processing
        test_data = QoETextDataset(x_test, y_test)
        test_loader = data_utils.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        model.eval()
        test_loss = 0
        correct = 0
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
            test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        result = result + 100. * correct / len(test_loader.dataset)
        logging.info('The {}th results: {}'.format(i+1, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset))))

    logging.info('Average = {}'.format(result / 10.0))
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))



    # model = Embedding()
