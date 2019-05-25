#!/usr/bin/env python3
import numpy as np

length = 200
x_tr_pos = np.load(f'./data/AMP_Scanner/AMP.tr.seq_{length}_x.npy')
x_va_pos = np.load(f'./data/AMP_Scanner/AMP.eval.seq_{length}_x.npy')
x_te_pos = np.load(f'./data/AMP_Scanner/AMP.te.seq_{length}_x.npy')
x_tr_neg = np.load(f'./data/AMP_Scanner/DECOY.tr.seq_{length}_x.npy')
x_va_neg = np.load(f'./data/AMP_Scanner/DECOY.eval.seq_{length}_x.npy')
x_te_neg = np.load(f'./data/AMP_Scanner/DECOY.te.seq_{length}_x.npy')

x_tr = np.concatenate((x_tr_pos, x_tr_neg))
x_va = np.concatenate((x_va_pos, x_va_neg))
x_te = np.concatenate((x_te_pos, x_te_neg))

y_tr = np.concatenate(([1]*len(x_tr_pos), [0]*len(x_tr_neg)))
y_va = np.concatenate(([1]*len(x_va_pos), [0]*len(x_va_neg)))
y_te = np.concatenate(([1]*len(x_te_pos), [0]*len(x_te_neg)))

np.save(f'./data/AMP_Scanner/tr.seq_{length}_x.npy', x_tr)
np.save(f'./data/AMP_Scanner/va.seq_{length}_x.npy', x_va)
np.save(f'./data/AMP_Scanner/te.seq_{length}_x.npy', x_te)

np.save(f'./data/AMP_Scanner/tr.seq_{length}_y.npy', y_tr)
np.save(f'./data/AMP_Scanner/va.seq_{length}_y.npy', y_va)
np.save(f'./data/AMP_Scanner/te.seq_{length}_y.npy', y_te)
