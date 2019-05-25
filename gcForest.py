#!/usr/bin/env python3
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, matthews_corrcoef

def subseq(seq, size=50, strides=1):
    return [seq[i:i+size] for i in range(0, int((len(seq)-size+1)/strides), strides)]

def multi_grained(x, y=None, size=50, strides=1):
    sub_x, sub_y = [], []
    for i in range(len(x)):
        sub = subseq(x[i], size, strides=1)
        sub_x.append(sub)
        if y is not None:
            sub_y.append([y[i]]*len(sub))

    print(np.array(sub_x).shape)
    return np.array(sub_x), np.array(sub_y)

def rf(x, y, n_estimators=100, depth=None):
    x = np.array([i for j in x for i in j])
    y = y.flatten()
    print(np.array(x).shape)
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=depth).fit(x, y)

def concat_proba(x, clfs, size=50, strides=1, grained=True):
    _x, _ = multi_grained(x, size=size, strides=strides)
    print(clfs[0].predict_proba(_x).shape)
    exit()
    probas = []
    for i in x:
        if grained:
            _x, _ = multi_grained([i], size=size, strides=strides) #151x50
        else:
            _x = [i]
        proba = np.array([list(clf.predict_proba(_x).flatten()) for clf in clfs]).flatten()
        probas.append(proba)

    return np.array(probas)

def gr_trees(x, y, windows):
    trees = {}
    for w in windows:
        grained_x, grained_y = multi_grained(x, y, size=w, strides=1)
        trees[f'rf_{w}_1'] = rf(grained_x, grained_y, n_estimators=10, depth=3)
        trees[f'rf_{w}_2'] = rf(grained_x, grained_y, n_estimators=10, depth=None)

    return trees

def gr_probas(x, trees, windows):
    probas = {}
    for w in windows:
        clfs = [trees[f'rf_{w}_{i+1}'] for i in range(2)]
        probas[f'{w}'] = concat_proba(x, clfs, size=w, strides=1)

    return probas

def ca_trees(x, y, window, trees):
    trees[f'rf_{window}_1'] = rf(x, y, n_estimators=10, depth=3)
    trees[f'rf_{window}_2'] = rf(x, y, n_estimators=10, depth=3)
    trees[f'rf_{window}_3'] = rf(x, y, n_estimators=10, depth=None)
    trees[f'rf_{window}_4'] = rf(x, y, n_estimators=10, depth=None)

    return trees

def ca_probas(x, trees, window, probas):
    clfs = [trees[f'rf_{window}_{i+1}'] for i in range(4)]
    probas[f'{window}'] = concat_proba(x, clfs, grained=False)

    return probas

if '__main__' == __name__:
    length = 200
    x_tr = np.load(f'./data/AMP_Scanner/tr.seq_{length}_x.npy')
    x_va = np.load(f'./data/AMP_Scanner/va.seq_{length}_x.npy')
    x_te = np.load(f'./data/AMP_Scanner/te.seq_{length}_x.npy')

    y_tr = np.load(f'./data/AMP_Scanner/tr.seq_{length}_y.npy')
    y_va = np.load(f'./data/AMP_Scanner/va.seq_{length}_y.npy')
    y_te = np.load(f'./data/AMP_Scanner/te.seq_{length}_y.npy')

    x_tr, y_tr = shuffle(x_tr, y_tr)
    x_va, y_va = shuffle(x_va, y_va)
    x_te, y_te = shuffle(x_te, y_te)

    windows = [int(length/16), int(length/8), int(length/4)]
    grained_trees = gr_trees(x_tr, y_tr, windows)
    print('-------')
    grained_probas = gr_probas(x_tr, grained_trees, windows)


    cascaded_trees, cascaded_probas = {}, {}
    for i in range(len(windows)):
        if i != 0:
            ca_input = np.concatenate((grained_probas[f'{windows[i]}'], cascaded_probas[f'{windows[i-1]}']), axis=1)
        else:
            ca_input = grained_probas[f'{windows[i]}']

        cascaded_trees = ca_trees(ca_input, y_tr, windows[i], cascaded_trees)
        cascaded_probas = ca_probas(ca_input, cascaded_trees, windows[i], cascaded_probas)

    ### test
    te_grained_probas = gr_probas(x_te, grained_trees, windows)
    te_cascaded_probas = {}
    for i in range(len(windows)):
        if i != 0:
            te_ca_input = np.concatenate((te_grained_probas[f'{windows[i]}'], te_cascaded_probas[f'{windows[i-1]}']), axis=1)
        else:
            te_ca_input = te_grained_probas[f'{windows[i]}']

        te_cascaded_probas = ca_probas(te_ca_input, cascaded_trees, windows[i], te_cascaded_probas)


    pred = [i.reshape(4, 2) for i in te_cascaded_probas[f'{windows[-1]}']]
    pred = [sum(i) for i in pred]
    ans = [np.argmax(i) for i in pred]

    print('acc: ', accuracy_score(y_te, ans))
    print('mcc: ', matthews_corrcoef(y_te, ans))
