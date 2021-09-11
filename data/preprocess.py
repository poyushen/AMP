#!/usr/bin/env python3
import numpy as np
import os
import pickle

proteins = 'ACDEFGHIKLMNPQRSTVWY'

def label_encode(seq):
    return np.array([proteins.index(i)+1 for i in list(seq)])


def gen_seq(data, file, label):
    with open(f'./{data}/fa/{file}', 'r') as f:
        raw = f.readlines()

    x = [raw[i].replace('\n', '') for i in range(1, len(raw), 2) if len(raw[i].replace('\n', ''))<=100]

    x = [label_encode(i) for i in x]
    y = [label]*len(x)

    print(f'There are {len(x)} sequences')
    ans = {'x': np.array(x), 'y': y}

    pickle.dump(ans, open(f'./{data}/seq/{file.split(".")[0]}.pkl', 'wb'))


if '__main__' == __name__:
    for folder in ['benchmark', 'scanner']:
        if not os.path.exists(f'./{folder}/seq/'):
            os.makedirs(f'./{folder}/seq/')

        for file in os.listdir(f'./{folder}/fa/'):
            label = 0 if 'neg' in file else 1
            gen_seq(folder, file, label)

