#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 13/9/17
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : min_max.py


from sklearn import preprocessing
import pickle

with open('data/embedding_data.pkl', 'rb') as f:
    data = pickle.load(f)
    x, y = data[0], data[1]
print x[..., [53]]

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(x[..., [53]])
x[..., [53]] = X_train_minmax

with open('data/embedding_data_new.pkl', 'wb') as f:
    pickle.dump([x, y], f)
print [x, y]
