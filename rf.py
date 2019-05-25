#!/usr/bin/env python3
import numpy as np
import random

from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score, classification_report, make_scorer, cohen_kappa_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from xgboost import XGBClassifier

from preprocess import shuffle_data

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

if '__main__' == __name__:
    x = np.load('./data/AmPEP/dis_x.npy')
    y = np.load('./data/AmPEP/dis_y.npy')

    results = {}
    indicators = ['sn', 'sp', 'acc', 'mcc', 'auc_roc', 'kappa']
    for i in indicators:
        results[i] = []

    def sn(y_true, y_pred): return recall_score(y_true, y_pred, average=None)[1]
    def sp(y_true, y_pred): return recall_score(y_true, y_pred, average=None)[0]

    scoring = {
        'sn': make_scorer(sn),
        'sp': make_scorer(sp),
        'acc': 'accuracy',
        'mcc': make_scorer(matthews_corrcoef),
        'auc_roc': 'roc_auc',
        'kappa': make_scorer(cohen_kappa_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score)
    }


    for i in range(20):
        print(i)
        x, y = shuffle_data(x, y, 1, i)

        clf = RandomForestClassifier(n_estimators=100)
        # clf = XGBClassifier(n_estimators=200, max_depth=5)

        result = cross_validate(clf, x, y, cv=10, scoring=scoring)

        for i in indicators:
            results[i].append(np.mean(result['test_%s' %i]))

    for i in indicators:
        print(f'{i}: {np.mean(results[i])}')
