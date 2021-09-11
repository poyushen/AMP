#!/usr/bin/env python3
import numpy as np
import pickle
import torch

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def gen_xy(tr_pos, tr_neg, te_pos, te_neg, va_pos=None, va_neg=None):
    def concat(x1, x2):
        return np.concatenate((x1, x2), axis=0)

    tr_x = concat(tr_pos['x'], tr_neg['x'])
    tr_y = concat(tr_pos['y'], tr_neg['y'])
    te_x = concat(te_pos['x'], te_neg['x'])
    te_y = concat(te_pos['y'], te_neg['y'])

    if va_pos is not None and va_neg is not None:
        va_x = concat(va_pos['x'], va_neg['x'])
        va_y = concat(va_pos['y'], va_neg['y'])

        return tr_x, tr_y, va_x, va_y, te_x, te_y

    return tr_x, tr_y, te_x, te_y


def load_data(path, file):
    with open(f'{path}/{file}.pkl', 'rb') as f:
        data = pickle.load(f)
    f.close()

    return data


def load_benchmark(mode='onehot', seed=0):
    path = './data/benchmark/seq'
    tr_pos = load_data(path, 'tr_pos')
    tr_neg = load_data(path, 'tr_neg')
    te_pos = load_data(path, 'te_pos')
    te_neg = load_data(path, 'te_neg')

    tr_x, tr_y, te_x, te_y = gen_xy(tr_pos, tr_neg, te_pos, te_neg)
    if 'raw' == mode:
        return tr_x, tr_y, te_x, te_y

    tr_x, te_x = pad(tr_x), pad(te_x)

    if 'onehot' == mode:
        tr_x, te_x = onehot(tr_x), onehot(te_x)

    tr_x, tr_y, te_x, te_y = shuffle_data(tr_x, tr_y, te_x, te_y, seed=seed)
    tr_x, va_x, tr_y, va_y = train_test_split(tr_x, tr_y, test_size=0.2, random_state=seed)
    tr_x, tr_y, va_x, va_y, te_x, te_y = np2torch(tr_x, tr_y, va_x, va_y, te_x, te_y, mode)

    return tr_x, tr_y, va_x, va_y, te_x, te_y


def load_AMP_Scanner(mode='onehot', seed=0):
    path = './data/scanner/seq'
    tr_pos = load_data(path, 'tr_pos')
    tr_neg = load_data(path, 'tr_neg')
    va_pos = load_data(path, 'va_pos')
    va_neg = load_data(path, 'va_neg')
    te_pos = load_data(path, 'te_pos')
    te_neg = load_data(path, 'te_neg')

    tr_x, tr_y, va_x, va_y, te_x, te_y = gen_xy(tr_pos, tr_neg, te_pos, te_neg, va_pos, va_neg)
    if 'raw' == mode:
        tr_x = np.concatenate((tr_x, va_x), axis=0)
        tr_y = np.concatenate((tr_y, va_y), axis=0)

        return tr_x, tr_y, te_x, te_y

    tr_x, va_x, te_x = pad(tr_x), pad(va_x), pad(te_x)
    if 'onehot' == mode:
        tr_x, va_x, te_x = onehot(tr_x), onehot(va_x), onehot(te_x)

    tr_x, tr_y, va_x, va_y, te_x, te_y = shuffle_data(tr_x, tr_y, te_x, te_y, va_x, va_y, seed=seed)
    tr_x, tr_y, va_x, va_y, te_x, te_y = np2torch(tr_x, tr_y, va_x, va_y, te_x, te_y, mode)

    return tr_x, tr_y, va_x, va_y, te_x, te_y


def np2torch(tr_x, tr_y, va_x, va_y, te_x, te_y, mode):
    def cvt(x):
        return torch.from_numpy(x)

    def cvt_long(x):
        return torch.from_numpy(x).long()

    if 'onehot' == mode:
        return cvt(tr_x), cvt(tr_y), cvt(va_x), cvt(va_y), cvt(te_x), cvt(te_y)
    else:
        return cvt_long(tr_x), cvt_long(tr_y), cvt_long(va_x), cvt_long(va_y), cvt_long(te_x), cvt_long(te_y)


def onehot(x):
    return np.array([to_categorical(i, num_classes=21) for i in x])


def pad(x):
    return pad_sequences(x, maxlen=100, padding='pre', truncating='post')


def shuffle_data(tr_x, tr_y, te_x, te_y, va_x=None, va_y=None, seed=0):
    tr_x, tr_y = shuffle(tr_x, tr_y, random_state=seed)
    te_x, te_y = shuffle(te_x, te_y, random_state=seed)

    if va_x is not None and va_y is not None:
        va_x, va_y = shuffle(va_x, va_y, random_state=seed)
        return tr_x, tr_y, va_x, va_y, te_x, te_y

    return tr_x, tr_y, te_x, te_y


