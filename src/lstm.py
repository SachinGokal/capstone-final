import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# General Model LSTM
class LSTM_Score_Predictor():

  def __init__(self):
    pass

  def build_model(self, first_layer_size=20):
    model = Sequential()

    model.add(LSTM(first_layer_size, return_sequences=True))
    model.add(Activation("tanh"))
    model.add(Dropout(0.2))

    model.add(LSTM(30, return_sequences=True))
    model.add(Activation("tanh"))
    model.add(Dropout(0.1))

    model.add(LSTM(40, return_sequences=False))
    model.add(Activation("tanh"))

    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam',
                  loss='mse')

    self.model = model

  def fit(self, X_train, y_train, batch_size=500, epochs=15):
    self.build_model()

    # input must be [samples, time steps, features]
    X = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

    return self.model.fit(X, y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=False)

  def predict(self, X_test):
    X = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    return self.model.predict(X)


# Category Model LSTM

class LSTM_Score_Predictor():

  def __init__(self):
    pass

  def build_model(self, first_layer_size=30):
    model = Sequential()

    model.add(LSTM(first_layer_size, return_sequences=False))
    model.add(Activation("tanh"))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam',
                  loss='mse')

    self.model = model

  def fit(self, X_train, y_train, batch_size=200, epochs=15):
    self.build_model()

    # input must be [samples, time steps, features]
    X = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

    return self.model.fit(X, y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=False)

  def predict(self, X_test):
    X = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    return self.model.predict(X)
