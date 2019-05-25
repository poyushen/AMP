#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Permute, BatchNormalization, Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, LSTM, Input, GlobalAveragePooling1D, multiply, Flatten, merge
from keras.models import Sequential, Model, load_model
from keras.regularizers import l1, l2
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from preprocess import shuffle_data

length = 200

def attention(inputs):
    input_dim = int(inputs.shape[2])
    time_step = 50
    a = Permute((2, 1))(inputs)
    a = Dense(time_step, activation='softmax')(a)

    a_probas = Permute((2, 1))(a)
    output = merge([inputs, a_probas], mode='mul')
    return output

def model_with_attention():
    input_x = Input(shape=(length, ))
    x = Embedding(input_dim=22, output_dim=128, input_length=length)(input_x)
    x = Conv1D(filters=64, kernel_size=16, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=5)(x)
    x = LSTM(100, use_bias=True, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    x = Dense(50, activation='relu')(x)
    att_probas = Dense(50, activation='softmax')(x)
    x = merge([x, att_probas], output_shape=50, mode='mul')
    x = Dropout(0.5)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_x, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def model_with_attention_lstm():
    input_x = Input(shape=(length, ))
    x = Embedding(input_dim=22, output_dim=64, input_length=length)(input_x)

    x = Conv1D(filters=32, kernel_size=8, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)

    x = Conv1D(filters=16, kernel_size=4, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)

    x = LSTM(64, use_bias=True, return_sequences=True)(x)
    x = Dropout(0.5)(x)

    x = LSTM(32, use_bias=True, return_sequences=True)(x)
    x = Dropout(0.5)(x)

    x = LSTM(16, use_bias=True, return_sequences=True)(x)
    x = Dropout(0.5)(x)

    x = attention(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)

    att_probas = Dense(100, activation='softmax')(x)
    x = merge([x, att_probas], output_shape=100, mode='mul')
    x = Dropout(0.5)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_x, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def model_with_attention_lstm_onehot():
    input_x = Input(shape=(length, 22))
    # x = Embedding(input_dim=22, output_dim=128, input_length=length)(input_x)
    x = Conv1D(filters=64, kernel_size=9, padding='same', activation='relu')(input_x)
    x = MaxPooling1D(pool_size=5)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(filters=32, kernel_size=9, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=5)(x)
    x = Dropout(0.3)(x)
    x = LSTM(100, use_bias=True, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = attention(x)
    x = Flatten()(x)
    x = Dense(50, activation='relu')(x)
    att_probas = Dense(50, activation='softmax')(x)
    x = merge([x, att_probas], output_shape=50, mode='mul')
    x = Dropout(0.5)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_x, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


if '__main__' == __name__:
    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session = tf.Session(config=config)

    # x = np.load(f'./data/AmPEP/seq_{length}_x.npy')
    # y = np.load(f'./data/AmPEP/seq_y.npy')
    #
    # x, y = shuffle_data(x, y, 1, 1)
    # x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2)

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

    # x_tr = np.array([to_categorical(i, num_classes=22) for i in x_tr])
    # x_va = np.array([to_categorical(i, num_classes=22) for i in x_va])
    # x_te = np.array([to_categorical(i, num_classes=22) for i in x_te])

    model = model_with_attention_lstm()
    model.summary()

    esp = EarlyStopping(patience=5)
    cpt = ModelCheckpoint('model.h5', save_best_only=True)
    model.fit(x_tr, y_tr, epochs=100, batch_size=64, validation_data=(x_va, y_va), callbacks=[esp, cpt])

    proba = model.predict(x_te)
    pred = [1 if i>0.5 else 0 for i in proba]
    print(f'acc: {accuracy_score(y_te, pred)}')
    print(f'mcc: {matthews_corrcoef(y_te, pred)}')
    print(f'auc: {roc_auc_score(y_te, proba)}')
