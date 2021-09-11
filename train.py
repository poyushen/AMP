#!/usr/bin/env python3
import argparse
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from scipy.special import softmax
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from torch.autograd import Variable

from util.load import load_benchmark, load_AMP_Scanner
from util.net import SENet, ConcatNet, ScannerNet
from util.earlystopping import EarlyStopping

import warnings
warnings.filterwarnings('ignore')


def build_model(model):
    if 'SENet' == model:
        net = SENet()

    if 'SENet_nopool' == model:
        net = SENet(pool=False)

    if 'ConcatNet' == model:
        net = ConcatNet()

    if 'ScannerNet' == model:
        net = ScannerNet()

    if torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark = True

    return net


def data_for_net(x, y, requires_grad=False):
    x = to_var(x, requires_grad=requires_grad)
    y = to_var(y, requires_grad=requires_grad)

    return x, y


def data_loader(x, y, batch_size=32):
    dataset = data.TensorDataset(x, y)

    return data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def get_loss(net, x, y, reduction='mean'):
    out = net(x)
    loss = nn.CrossEntropyLoss(reduction=reduction)(out, y)

    return out, loss


def performance(net, x, y):
    x, y = data_for_net(x, y)
    out, _ = get_loss(net, x, y)

    proba = softmax(out.cpu().data.numpy(), axis=1)
    pred = [1 if i>=0.5 else 0 for i in proba[:, 1]]

    y = y.cpu().data.numpy()
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, proba[:, 1])
    mcc = matthews_corrcoef(y, pred)

    return acc, auc, mcc, pred


def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()

    return Variable(x, requires_grad=requires_grad)


def train(tr_x, tr_y, va_x, va_y, model):
    tr_x, tr_y = data_for_net(tr_x, tr_y)
    va_x, va_y = data_for_net(va_x, va_y)

    net = build_model(model)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    esp = EarlyStopping(patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3, verbose=True)

    batch_size = 16
    epochs = 100
    tr_losses = []
    va_losses = []

    for epoch in range(epochs):
        net.train()
        for _x, _y in data_loader(tr_x, tr_y, batch_size=batch_size):
            out, loss = get_loss(net, _x, _y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_losses.append(loss.item())

        net.eval()
        for _x, _y in data_loader(va_x, va_y, batch_size=batch_size):
            out, loss = get_loss(net, _x, _y)

            va_losses.append(loss.item())

        tr_loss = np.mean(tr_losses)
        va_loss = np.mean(va_losses)
        scheduler.step(va_loss)

        tr_losses, va_losses = [], []

        print(f'{epoch+1} / {epochs}')
        print(f'tr_loss: {tr_loss: .5f}   va_loss: {va_loss: .5f}')

        esp(va_loss, net)
        if esp.early_stop:
            break

    return net


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='benchmark/scanner')
    parser.add_argument('--model', type=str, default='model', help='SENet/SENet_nopool/ConcatNet/Scanner')

    args = parser.parse_args()
    dataset = args.data
    model = args.model

    accs, aucs, mccs, preds = [], [], [], []
    iters = 20

    path = f'./results/{dataset}/{model}/'
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(iters):
        torch.cuda.empty_cache()

        # for benchmark
        if 'benchmark' == dataset:
            if 'ScannerNet' == model:
                tr_x, tr_y, va_x, va_y, te_x, te_y = load_benchmark(mode='scanner')
            else:
                tr_x, tr_y, va_x, va_y, te_x, te_y = load_benchmark(mode='onehot')

        # for AMPScanner
        elif 'scanner' == dataset:
            if 'ScannerNet' == model:
                tr_x, tr_y, va_x, va_y, te_x, te_y = load_AMP_Scanner(mode='scanner')
            else:
                tr_x, tr_y, va_x, va_y, te_x, te_y = load_AMP_Scanner(mode='onehot')

        print(f'=== epoch {i+1}/{iters} ===')

        net = train(tr_x, tr_y, va_x, va_y, model)
        acc, auc, mcc, pred = performance(net, te_x, te_y)

        accs.append(acc)
        aucs.append(auc)
        mccs.append(mcc)
        preds.append(confusion_matrix(te_y, pred).ravel())

        torch.save(net, f'./{path}/model_{i}.pkl')
    print()

    f = open(f'{path}/results.log', 'w')

    for i in range(len(accs)):
        print(f'ACC: {accs[i]: .3f}  AUC: {aucs[i]: .3f}  MCC: {mccs[i]: .3f}')
        f.write(f'ACC: {accs[i]: .3f}  AUC: {aucs[i]: .3f}  MCC: {mccs[i]: .3f}\n')
    print()
    f.write(f'\nACC: {np.mean(accs): .3f}  AUC: {np.mean(aucs): .3f}  MCC: {np.mean(mccs): .3f}\n')
    print(f'ACC: {np.mean(accs): .3f}  AUC: {np.mean(aucs): .3f}  MCC: {np.mean(mccs): .3f}')

    idx = np.argmax(mccs)
    print(f'ACC: {accs[idx]: .3f}  AUC: {aucs[idx]: .3f}  MCC: {mccs[idx]: .3f}')
    f.write(f'ACC: {accs[idx]: .3f}  AUC: {aucs[idx]: .3f}  MCC: {mccs[idx]: .3f}\n')

    print(f'ACC: {np.std(accs): .3f}  AUC: {np.std(aucs): .3f}  MCC: {np.std(mccs): .3f}')
    f.write(f'ACC: {np.std(accs): .3f}  AUC: {np.std(aucs): .3f}  MCC: {np.std(mccs): .3f}')

    f.close()
