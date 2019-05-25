#!/usr/bin/env python3
import math
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.utils import shuffle
from xgboost import XGBClassifier

class gcForest():
    def __init__(self, windows, n_gr_trees, n_ca_trees):

        self.windows = windows
        self.n_windows = len(windows)
        self.n_gr_trees = n_gr_trees
        self.n_ca_trees = n_ca_trees

        self.n_layers = 0

        self.gr_trees = {}
        self.ca_trees = {}

    def subseq(self, seq, size, strides=1):
        n = math.ceil((len(seq)-size+1)/strides)
        end = (n - 1) * strides + 1
        return [seq[i:i+size] for i in range(0, end, strides)]

    def multi_grained(self, x, y, size, strides=1):
        sub_x, sub_y = [], []
        for i in range(len(x)):
            sub = self.subseq(x[i], size, strides=1)
            sub_x.extend(sub)
            if y is not None:
                sub_y.extend([y[i]]*len(sub))

        return np.array(sub_x), np.array(sub_y)

    def rf(self, x, y, params):
        return RandomForestClassifier(**params).fit(x, y)

    def xgb(self, x, y, params):
        return XGBClassifier(**params).fit(x, y)

    def concat_proba(self, x, clfs, size=None, strides=None, grained=False):
        probas = []
        for i in x:
            _x = self.multi_grained([i], y=None, size=size, strides=strides)[0] if grained else [i]
            proba = np.array([list(clf.predict_proba(_x).flatten()) for clf in clfs]).flatten()
            probas.append(proba)

        return np.array(probas)

    def gr_models(self, x, y):
        '''
        self.tr_gr_trees = {
            12: [rf, cmplt_rf, rf, cmplt_rf, ......],
            25: [rf, cmplt_rf, rf, cmplt_rf, ......],
            50: [rf, cmplt_rf, rf, cmplt_rf, ......]
        }
        '''
        for w in self.windows:
            self.gr_trees[w] = []
            grained_x, grained_y = self.multi_grained(x, y, size=w, strides=1)
            for i in range(self.n_gr_trees):
                # self.gr_trees[w].append(self.rf(grained_x, grained_y, {'n_estimators': 100, 'max_depth': 3, 'n_jobs': 4}))
                # self.gr_trees[w].append(self.rf(grained_x, grained_y, {'n_estimators': 10, 'max_depth': None, 'n_jobs': 4}))
                self.gr_trees[w].append(self.xgb(grained_x, grained_y, {'n_estimators': 120, 'max_depth': 3, 'n_jobs': 4}))
                self.gr_trees[w].append(self.xgb(grained_x, grained_y, {'n_estimators': 120, 'max_depth': 5, 'n_jobs': 4}))

    def gr_probas(self, x):
        '''
        self.tr_gr_probas = {
            12: array 1424x756
            25: array 1424x704,
            50: array 1424x604
        }
        '''
        probas = {}
        for w in self.windows:
            clfs = self.gr_trees[w]
            probas[w] = self.concat_proba(x, clfs, size=w, strides=1, grained=True)

        return probas

    def ca_models(self, y):
        '''
        self.tr_ca_probas = {
            1: [.....],
            2: [.....],
            .....
        }
        '''
        try:
            print(self.tr_ca_probas[self.n_layers-1])
        except:
            pass
        self.tr_ca_probas[self.n_layers] = [] # 3x8
        self.ca_trees[self.n_layers] = {}
        for i, w in enumerate(self.windows):
            self.ca_trees[self.n_layers][w] = []
            if i==0:
                if self.n_layers == 0:
                    ca_input = self.tr_gr_probas[w]
                else:
                    ca_input = np.concatenate((self.tr_gr_probas[w], self.tr_ca_probas[self.n_layers-1][-1]), axis=1)
            else:
                ca_input = np.concatenate((self.tr_gr_probas[w], self.tr_ca_probas[self.n_layers][-1]), axis=1)

            for j in range(self.n_ca_trees): ## 2
                # self.ca_trees[self.n_layers][w].append(self.rf(ca_input, y, {'n_estimators': 100, 'max_depth':3, 'n_jobs': 4}))
                # self.ca_trees[self.n_layers][w].append(self.rf(ca_input, y, {'n_estimators': 10, 'max_depth':None, 'n_jobs': 4}))
                self.ca_trees[self.n_layers][w].append(self.xgb(ca_input, y, {'n_estimators': 120, 'max_depth':3, 'n_jobs': 4}))
                self.ca_trees[self.n_layers][w].append(self.xgb(ca_input, y, {'n_estimators': 120, 'max_depth':5, 'n_jobs': 4}))
            self.tr_ca_probas[self.n_layers].append(self.concat_proba(ca_input, self.ca_trees[self.n_layers][w]))

        self.n_layers += 1

    def ca_probas(self, gr_probas):
        probas = {}
        for n in range(self.n_layers):
            probas[n] = []
            for i, w in enumerate(self.windows):
                if i==0:
                    if n==0:
                        ca_input = gr_probas[w]
                    else:
                        ca_input = np.concatenate((gr_probas[w], probas[n-1][-1]), axis=1)
                else:
                    ca_input = np.concatenate((gr_probas[w], probas[n][-1]), axis=1)
                probas[n].append(self.concat_proba(ca_input, self.ca_trees[n][w]))

        return probas

    def fit(self, x, y): # tr_x, tr_y
        self.gr_models(x, y)
        self.tr_gr_probas = self.gr_probas(x)
        self.tr_ca_probas = {}
        self.ca_models(y)

    def predict(self, x):
        gr_probas = self.gr_probas(x)
        ca_probas = self.ca_probas(gr_probas)

        return ca_probas

    def evaluate(self, x_va, y_va, y_tr):
        best_mcc = 0
        count = 0
        repeat = 0
        gr_va_probas = self.gr_probas(x_va)
        while count <= 10 and repeat <= 3:
            count += 1
            print(count)
            probas = self.ca_probas(gr_va_probas)[self.n_layers-1][-1]
            probas = [i.reshape(2*self.n_ca_trees, 2) for i in probas]
            probas = [sum(i) for i in probas]
            pred = [np.argmax(i) for i in probas]
            mcc = matthews_corrcoef(y_va, pred)
            print(mcc)
            if mcc > best_mcc:
                if mcc == best_mcc:
                    repeat += 1
                best_mcc = mcc
                self.ca_models(y_tr)
            else:
                self.n_layers -= 1
                del self.ca_trees[self.n_layers]
                print(self.ca_trees.keys())
                break

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

    # x_tr, y_tr = x_tr[:50], y_tr[:50]
    # x_va, y_va = x_va[:50], y_va[:50]
    # x_te, y_te = x_te[:50], y_te[:50]

    windows = [int(length/16), int(length/8), int(length/4)]
    gc = gcForest(windows=windows, n_gr_trees=1, n_ca_trees=2)
    gc.fit(x_tr, y_tr)

    gc.evaluate(x_va, y_va, y_tr)

    probas = gc.predict(x_te)[gc.n_layers-1][-1]
    probas = [i.reshape(2*gc.n_ca_trees, 2) for i in probas]
    probas = [sum(i) for i in probas]
    pred = [np.argmax(i) for i in probas]

    print('acc: ', accuracy_score(y_te, pred))
    print('mcc: ', matthews_corrcoef(y_te, pred))
