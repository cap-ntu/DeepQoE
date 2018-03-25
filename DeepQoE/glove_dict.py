#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 28/8/17
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : glove_dict.py

from __future__ import division
from __future__ import print_function

import os
from torchtext import vocab
from collections import Counter

from config import cfg


def get_dicts_base(words=['hello', 'world'], glove_path='../models/GloVe/glove.6B.50d.txt'):
    if not os.path.isfile(glove_path):
        print("Please use script to download pre-trained glove models...")
        return False
    else:
        vectors = {}
        dicts = {}
        with open(glove_path, 'r') as f:
            for line in f:
                vals = line.rstrip().split(' ')
                vectors[vals[0]] = [float(x) for x in vals[1:]]
        for i in words:
            dicts[i] = vectors[i]
        return dicts


def get_dicts(words=['hello', 'world'], glove='glove.6B.50d'):
    c = Counter(words)
    v = vocab.Vocab(c, vectors=glove)
    dicts = {}
    for i in words:
        dicts[i] = v.vectors.numpy()[v.stoi[i]]
    return dicts


if __name__ == '__main__':
    get_dicts()
