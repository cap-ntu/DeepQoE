#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 28/8/17
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : config.py
import argparse
from easydict import EasyDict as edict

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


__C = edict()
cfg = __C

__C.EMBEDDING_DATA = 'data/train_test_glove.6B.50d.pkl'
__C.MOS_DATA = 'data/MOS_train_data.pkl'
__C.JND_DATA = 'data/videos_c3d_features_labels.pkl'
__C.NFLX_DATA = 'data/LIVE_NFLX_PublicData_VideoATLAS_Release'

__C.TRAIN_RATIO = 0.9

__C.DIMENSION = 60
__C.LOG = 'bit'


__C.MODEL_SAVE_TEXT = 'models/GloVe/QoE_score.pt'
__C.MODEL_SAVE_MOS = 'models/GloVe/QoE_MOS_sport.pt'
__C.MODEL_SAVE_VIDEO = 'models/C3D/QoE_JND.pt'

__C.CLASSIFIER_NAME = ['SVM', 'Decision Tree', 'Random Forest', 'AdaBoost', 'Naive Bayes']
__C.CLASSIFIER = [svm.SVC(decision_function_shape='ovo'),
                   DecisionTreeClassifier(max_depth=5),
                   RandomForestClassifier(max_depth=5, n_estimators=10, max_features=3),
                   AdaBoostClassifier(),
                   GaussianNB()]


__C.POOLING_TYPE = 'mean'
__C.QUALITY_MODEL = 'STRRED'
__C.PREPROC = False
__C.FEATURE_NAMES = ["VQA", "R$_1$", "R$_2$", "M", "I"]
__C.DB_PATH = 'data/LIVE_NFLX_PublicData_VideoATLAS_Release/'
__C.TRAIN_TEST_NFLX = 'TrainingMatrix_LIVENetflix_1000_trials'

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='DeepQoE training')
    parser.add_argument('--model_number', type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='use GPU')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='selected a gpu')
    parser.add_argument('--classifier', type=int, default=1,
                        help='selected a classifier')
    return parser.parse_args(argv)
