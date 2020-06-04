import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error

import os
import cv2
import glob
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model

# Switch to classification?


class LSTM_Score_Predictor():

  def __init__(self):
    pass

  def build_model(self, first_layer_size=35):
    model = Sequential()
    model.add(LSTM(first_layer_size, return_sequences=True, stateful=False))
    model.add(LSTM(first_layer_size, return_sequences=False, stateful=False))
    model.add(Activation("tanh"))
    model.add(Dropout(0.2))

    # model.add(Dense(units=1))
    # model.add(Activation("linear"))
    # model.compile(optimizer='adam', loss='mse')

    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics='accuracy')

    self.model = model

  # batch size is 30 to train on roughly 30 frames at a time
  def fit(self, X_train, y_train, batch_size=30, epochs=20):
    self.build_model()

    # input must be [samples, time steps, features]
    X = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

    self.model.fit(X, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

  def predict(self, X_test):
    X = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    return self.model.predict(self.X)
