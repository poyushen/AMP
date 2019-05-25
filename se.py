#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, Flatten, Input, multiply, Embedding, LSTM
from keras.models import Model
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)

length = 200

def model():
    input_x = Input(shape=(length, ))
    embedding = Embedding(input_dim=21, output_dim=128, input_length=length)(input_x)

    cnn1_1 = Conv1D(64, kernel_size=16, strides=1, activation='relu', padding='same')(embedding)
    se1 = GlobalAveragePooling1D()(cnn1_1)
    se1 = Dense(64, activation='relu')(se1)
    se1 = multiply([cnn1_1, se1])
    se1 = Dropout(0.1)(se1)

    # cnn2_1 = Conv1D(32, kernel_size=8, strides=1, activation='relu', padding='valid')(se1)
    # se2 = GlobalAveragePooling1D()(cnn2_1)
    # se2 = Dense(32, activation = 'relu')(se2)
    # se2 = multiply([cnn2_1, se2])

    gap = GlobalAveragePooling1D()(se1)
    dense1 = Dense(1, activation='sigmoid')(gap)
    model = Model(inputs = input_x, outputs = dense1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if '__main__' == __name__:
    K.clear_session()

    x_tr = np.load(f'./data/{length}/x_train.npy')
    y_tr = np.load(f'./data/{length}/y_train.npy')
    x_va = np.load(f'./data/{length}/x_val.npy')
    y_va = np.load(f'./data/{length}/y_val.npy')
    x_te = np.load(f'./data/{length}/x_test.npy')
    y_te = np.load(f'./data/{length}/y_test.npy')

    # x_tr = np.array([to_categorical(i, num_classes=21) for i in x_tr])
    # x_va = np.array([to_categorical(i, num_classes=21) for i in x_va])
    # x_te = np.array([to_categorical(i, num_classes=21) for i in x_te])

    # y_tr = to_categorical(y_tr, num_classes=2)
    # y_va = to_categorical(y_va, num_classes=2)

    model = model()
    model.summary()
    esp = EarlyStopping(patience=5)
    model.fit(x_tr, y_tr, epochs=1, batch_size=64, validation_data=(x_va, y_va), callbacks=[esp])

    proba = model.predict(x_te)#[:, 1]
    pred = [1 if i>0.5 else 0 for i in proba]
    print('acc: ', accuracy_score(y_te, pred))
    print('mcc: ', matthews_corrcoef(y_te, pred))
    print('auc: ', roc_auc_score(y_te, proba))


