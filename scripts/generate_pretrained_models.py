#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/3/18 10:42 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : generate_pretrained_models.py

# References:
# 1) C. G. Bampis and A. C. Bovik, "Video ATLAS Software Release"
# URL: http://live.ece.utexas.edu/research/Quality/VideoATLAS_release.zip, 2016
# 2) C. G. Bampis and A. C. Bovik, "Learning to Predict Streaming Video QoE: Distortions, Rebuffering and Memory," under review

from __future__ import print_function

import os
import re
import numpy as np
import scipy.io as sio
import copy
from sklearn import preprocessing
from DeepQoE.config import cfg


def generate_streaming_data():

    db_files = os.listdir(cfg.DB_PATH)
    db_files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    Nvideos = len(db_files)

    pre_load_train_test_data_LIVE_Netflix = sio.loadmat('data/'+cfg.TRAIN_TEST_NFLX+'.mat')[cfg.TRAIN_TEST_NFLX]

    # randomly pick a trial out of the 1000
    nt_rand = np.random.choice(np.shape(pre_load_train_test_data_LIVE_Netflix)[1], 1)
    n_train = [ind for ind in range(0, Nvideos) if pre_load_train_test_data_LIVE_Netflix[ind, nt_rand] == 1]
    n_test = [ind for ind in range(0, Nvideos) if pre_load_train_test_data_LIVE_Netflix[ind, nt_rand] == 0]

    X = np.zeros((len(db_files), len(cfg.FEATURE_NAMES)))
    y = np.zeros((len(db_files), 1))

    feature_labels = list()
    for typ in cfg.FEATURE_NAMES:
        if typ == "VQA":
            feature_labels.append(cfg.QUALITY_MODEL + "_" + cfg.POOLING_TYPE)
        elif typ == "R$_1$":
            feature_labels.append("ds_norm")
        elif typ == "R$_2$":
            feature_labels.append("ns")
        elif typ == "M":
            feature_labels.append("tsl_norm")
        else:
            feature_labels.append("lt_norm")

    for i, f in enumerate(db_files):
        data = sio.loadmat(cfg.DB_PATH + f)
        for feat_cnt, feat in enumerate(feature_labels):
            X[i, feat_cnt] = data[feat]
        y[i] = data["final_subj_score"]

    X_train_before_scaling = X[n_train, :]
    X_test_before_scaling = X[n_test, :]
    y_train = y[n_train]
    y_test = y[n_test]

    if cfg.PREPROC:
        scaler = preprocessing.StandardScaler().fit(X_train_before_scaling)
        X_train = scaler.transform(X_train_before_scaling)
        X_test = scaler.transform(X_test_before_scaling)
    else:
        X_train = copy.deepcopy(X_train_before_scaling)
        X_test = copy.deepcopy(X_test_before_scaling)

    return X_train, y_train, X_test, y_test, feature_labels
