#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19/2/18 8:43 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : cal_corre.py

from __future__ import print_function
import torch
import numpy as np
from scripts.generate_pretrained_models import show_corre
from scipy.stats.stats import pearsonr
from scipy.stats import kendalltau
from scipy.stats import spearmanr
import torch.utils.data as data_utils
from DeepQoE.nets import *
from DeepQoE.config import cfg
from DeepQoE.data_loader import QoEMOSDataset
import pickle

with open(cfg.MOS_DATA, 'rb') as f:
    data = pickle.load(f)
    x = data[:, 0:data.shape[1] - 1]
    y = np.array(data[:, -1], np.float)

model = HybridRR()
model.cuda()
model.load_state_dict(torch.load(cfg.MODEL_SAVE_MOS))

test_data = QoEMOSDataset(x, y)
test_loader = data_utils.DataLoader(test_data, batch_size=72, shuffle=False)
model.eval()
for sample_batched in test_loader:
    x_1 = torch.autograd.Variable(sample_batched['glove'].cuda())
    x_2 = torch.autograd.Variable(sample_batched['res'].cuda())
    x_3 = torch.autograd.Variable(sample_batched['bitrate'].cuda())
    target = torch.autograd.Variable(sample_batched['label'].cuda())
    output, _ = model(x_1, x_2, x_3)

y_pred = output.data.cpu().numpy().squeeze()
print(y, y_pred)

show_corre(y[36:54], y_pred[36:54])

print("Pearson: {}".format(pearsonr(y, y_pred)))
print("Kendal: {}".format(kendalltau(y, y_pred)))
print("Spearman: {}".format(spearmanr(y, y_pred)))
