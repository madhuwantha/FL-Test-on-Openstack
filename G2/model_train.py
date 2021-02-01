from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import numpy as np
import pandas as pd
import glob
from os import path

batch_size = 15
epochs = 500

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)


def process_data():
    print("Processing data...")
    x = pd.read_csv('data/x_train.csv')
    y = pd.read_csv('data/y_train.csv')
    x_train = x.sample(frac=0.8, random_state=0)
    x_test = x.drop(x_train.index)

    y_train = y.sample(frac=0.8, random_state=0)
    y_test = y.drop(y_train.index)

    return x_train, x_test, y_train, y_test


def build_model(x_train, y_train, x_test, y_test, lr=0.0005):
    # if there is model update set its weights as model weights
    if path.exists("model_update/agg_model.h5"):
        print("Agg model exists...\nLoading model...")
        model = load_model("model_update/agg_model.h5", compile=False)
    else:
        print("No agg model found!\nBuilding model...")
        model = Sequential()
        model.add(Dense(units=200, activation='relu', input_shape=[92]))
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.7))
        model.add(Flatten())
        model.add(Dense(400, activation='relu'))
        model.add(Dropout(0.8))
        model.add(Dense(5, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=lr, momentum=0.9),
                      metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=50)
    mc = ModelCheckpoint('local_model/best_model.h5', monitor='val_accuracy', mode='max', verbose=0,
                         save_best_only=True)

    print(model.summary())
    print("Model train starting ......")
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[es, mc]
              )
    model = load_model('local_model/best_model.h5')

    return model


def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def save_local_model_update(model):
    mod = model.get_weights()
    np.save('local_model/mod', mod)
    print("Local model update written to local storage!")


def train():
    x_train, x_test, y_train, y_test = process_data()
    model = build_model(x_train, y_train, x_test, y_test)
    evaluate_model(model, x_test, y_test)
    save_local_model_update(model)
