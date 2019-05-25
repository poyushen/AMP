#!/usr/bin/env python3
import numpy as np
import random
from collections import Counter
from keras.utils import to_categorical

pos_path = './data/AMP_Scanner/tmp/DECOY.tr.fa'
neg_path = './data/AMP_Scanner/fa/nonAMP.fasta'

with open(pos_path, 'r') as f:
    pos_raw = f.readlines()
with open(neg_path, 'r') as f:
    neg_raw = f.readlines()

proteins = 'ACDEFGHIKLMNPQRSTVWY'
length = 200

properties = {
    'hydrophobicity': [
        'RKEDQN',  # Polar
        'GASTPHY', # Neutral
        'CLVIMEW',# Hydrophobic
    ],

    'vanderwaals_volume': [
        'GASTPD',  # Volume range 0-2.78
        'NVEQIL',  # Volume range 2.95-4.0
        'MHKFRYW', # Volume range 4.03-8.08
    ],

    'polarity': [
        'LIFWCMVY', # Polarity range 4.9-6.2
        'PATGS',    # Polarity range 8.0-9.2
        'HQRKNED',  # Polarity range 10.4-13
    ],

    'polarizability': [
        'GASDT',   # Polarizability value 0-0.108
        'CPNVQIL', # Polarizability value 0.128-0.186
        'KMHFRYW', # Polarizability value 0.219-0.409
    ],

    'charge': [
        'KR',               # Positive
        'ANCQGHILMFPSTWYV', # Neutral
        'DE',               # Negative
    ],

    'secondary_structure': [
        'EALMQKRH', # Helix
        'VIYCWFT',  # Strand
        'GNPSD',    # Coil
    ],

    'solvent_accessibility': [
        'ALFCGIVW', # Buried
        'PKQEND',   # Exposed
        'MPSTHY',   # Intermediate
    ]
}


def classes(acid, key):
    for i in range(3):
        if acid in properties[key][i]:
            return i

def convert(seq, key):
    return [classes(acid, key) for acid in list(seq)]

def distribution(seq, category):
    position = np.where(np.array(seq) == category)[0]
    n = len(position)
    if n == 0:
        return [0, 0, 0, 0, 0]
    idx = np.array([position[0], position[round(0.25*n)-1], position[round(0.5*n)-1], position[round(0.75*n)-1], position[-1]]) + 1

    return [i/len(seq) for i in idx]

def onehot(seq):
    tmp = [proteins.index(i) if i in proteins else 20 for i in list(seq)]
    tmp = tmp + [21]*(length-len(tmp)) if len(tmp)<length else tmp[:length]
    return tmp
    # return to_categorical(tmp, num_classes=21)

def shuffle_data(x, y, ratio, random_seed):
    x_pos, y_pos, x_neg, y_neg = x[y==1], y[y==1], x[y==0], y[y==0]

    neg = list(zip(x_neg, y_neg))
    random.Random(random_seed).shuffle(neg)
    x_neg, y_neg = map(np.array, zip(*neg))

    x = np.concatenate((x_pos, x_neg[:(len(x_pos)*ratio)]))
    y = np.concatenate((y_pos, y_neg[:(len(x_pos)*ratio)]))

    data = list(zip(x, y))
    random.Random(5).shuffle(data)
    x, y = map(np.array, zip(*data))

    return x, y

def dis_data():
    pos = [pos_raw[i].replace('\n', '') for i in range(1, len(pos_raw), 2)]
    # neg = [neg_raw[i].replace('\n', '') for i in range(1, len(neg_raw), 2)]
    # _len = [len(i) for i in neg]
    # ind = np.where(np.array(_len)<200)[0]
    # neg = list(np.array(neg)[ind])

    # x = pos + neg
    # y = [1]*len(pos) + [0]*len(neg)
    x = pos

    dis_x = []
    for _x in x:
        features = [distribution(convert(_x, key), i) for i in range(3) for key in sorted(properties.keys())]
        dis_x.append(np.array(features).flatten())

    # print(np.array(dis_x).shape)
    np.save('./data/AMP_Scanner/DECOY.tr.dis_x.npy', np.array(dis_x))
    # np.save('./data/AMP_Scanner/dis_y.npy', np.array(y))

def seq_data():
    pos = [pos_raw[i].replace('\n', '') for i in range(1, len(pos_raw), 2)]
    # neg = [neg_raw[i].replace('\n', '') for i in range(1, len(neg_raw), 2)]
    # _len = [len(i) for i in neg]
    # ind = np.where(np.array(_len)<200)[0]
    # neg = list(np.array(neg)[ind])

    # x = pos + neg
    x = pos
    x = [onehot(i) for i in x]
    # y = [1]*len(pos) + [0]*len(neg)
    # print(np.array(x).shape)

    np.save(f'./data/AMP_Scanner/DECOY.tr.seq_{length}_x.npy', np.array(x))
    # np.save('./data/AMP_Scanner/seq_y.npy', np.array(y))

if '__main__' == __name__:
    dis_data()
    seq_data()
