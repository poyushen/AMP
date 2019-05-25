#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import keras.layers.core as core
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Activation, BatchNormalization, Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, LSTM, Input, GlobalAveragePooling1D, multiply, Flatten, merge
from keras.models import Sequential, Model
from keras.regularizers import l1, l2
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from preprocess import shuffle_data

from attention import Attention, myFlatten

length = 200

def MultiCNN():
    input_row = 200
    input_col = 22

    input_x = Input(shape=(input_row, input_col))
    x = Conv1D(64, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=l1(0), padding='same')(input_x)
    x = Dropout(0.3)(x)
    x = Activation('relu')(x)
    x = Conv1D(32, kernel_size=5, kernel_initializer='he_normal', kernel_regularizer=l1(0), padding='same')(x)
    x = Activation('relu')(x)
    x_reshape = core.Reshape((x._keras_shape[2], x._keras_shape[1]))(x)

    x = Dropout(0.3)(x)
    x_reshape = Dropout(0.3)(x_reshape)

    decoder_x = Attention(hidden=10, activation='linear', init='he_normal', W_regularizer=l1(0.15))
    decoded_x = decoder_x(x)
    output_x = myFlatten(x._keras_shape[2])(decoded_x)

    decoder_xr = Attention(hidden=8, activation='linear', init='he_normal', W_regularizer=l1(2))
    decoded_xr = decoder_xr(x_reshape)
    output_xr = myFlatten(x_reshape._keras_shape[2])(decoded_xr)

    output = merge([output_x, output_xr], mode='concat')
    output = Dropout(0.5)(output)
    output = Dense(8, kernel_initializer='he_normal', activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(2, kernel_initializer='he_normal', activation='softmax')(output)

    cnn = Model(input_x, output)
    cnn.compile(loss='categorical_crosstropy', optimizer='adam', metrics=['accuracy'])

    return cnn


if '__main__' == __name__:
    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session = tf.Session(config=config)

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

    x_tr = np.array([to_categorical(i, num_classes=22) for i in x_tr])
    x_va = np.array([to_categorical(i, num_classes=22) for i in x_va])
    x_te = np.array([to_categorical(i, num_classes=22) for i in x_te])

    y_tr = to_categorical(y_tr, num_classes=2)
    y_va = to_categorical(y_va, num_classes=2)
    y_te = to_categorical(y_te, num_classes=2)

    model = MultiCNN()
    model.fit(x_tr, y_tr, epochs=100, batch_size=64, validation_data=(x_va, y_va), callbacks=[esp])

    proba = model.predict(x_te)
    pred = [1 if i>0.5 else 0 for i in proba]
    print(f'acc: {accuracy_score(y_te, pred)}')
    print(f'mcc: {matthews_corrcoef(y_te, pred)}')
    print(f'auc: {roc_auc_score(y_te, proba)}')
