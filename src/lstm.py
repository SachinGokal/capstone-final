import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

class LSTM_Score_Predictor():

  def __init__(self):
    pass

  def build_model(self, first_layer_size=100):
    model = Sequential()

    model.add(LSTM(first_layer_size, return_sequences=False))
    model.add(Activation("tanh"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam',
                  loss='mse')

    self.model = model

  def fit(self, X_train, y_train, batch_size=200, epochs=30):
    self.build_model()

    # input must be [samples, time steps, features]
    X = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

    return self.model.fit(X, y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=False)

  def predict(self, X_test):
    X = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    return self.model.predict(X)

    model = Sequential()

    model.add(LSTM(first_layer_size, return_sequences=False))
    model.add(Activation("tanh"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam',
                  loss='mse')

    X_for_fit = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    history = model.fit(X, y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=False)

def rmse(mse):
  return np.sqrt(mse)

# class LSTM_Score_Predictor():

#   def __init__(self):
#     pass

#   def build_model(self, first_layer_size=35):
#     model = Sequential()

#     model.add(Bidirectional(LSTM(1000, return_sequences=False)))
#     model.add(Activation("tanh"))
#     model.add(Dropout(0.5))

#     model.add(Dense(5))
#     model.add(Activation('softmax'))

#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam', metrics='accuracy')

#     self.model = model

#   # batch size is 30 to train on roughly 30 frames at a time
#   def fit(self, X_train, y_train, batch_size=2000, epochs=20):
#     self.build_model()

#     # input must be [samples, time steps, features]
#     X = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

#     self.model.fit(X, y_train, batch_size=batch_size,
#                    epochs=epochs, verbose=1, shuffle=False)

#   def predict(self, X_test):
#     X = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
#     return self.model.predict(self.X)

# GOLD
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
    model.add(Dropout(0.2))

    model.add(LSTM(40, return_sequences=False))
    model.add(Activation("tanh"))

    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam',
                  loss='mse')

    self.model = model

  def fit(self, X_train, y_train, batch_size=500, epochs=30):
    self.build_model()

    # input must be [samples, time steps, features]
    X = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

    return self.model.fit(X, y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=False)

  def predict(self, X_test):
    X = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    return self.model.predict(X)

class LSTM_Score_Predictor():

  def __init__(self):
    pass

  def build_model(self, first_layer_size=10):
    model = Sequential()

    model.add(LSTM(first_layer_size, return_sequences=True))
    model.add(Activation("tanh"))
    model.add(Dropout(0.3))

    model.add(LSTM(20, return_sequences=True))
    model.add(Activation("tanh"))
    model.add(Dropout(0.2))

    model.add(LSTM(30, return_sequences=False))
    model.add(Activation("tanh"))

    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam',
                  loss='mse')

    self.model = model

  def fit(self, X_train, y_train, batch_size=500, epochs=30):
    self.build_model()

    # input must be [samples, time steps, features]
    X = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

    return self.model.fit(X, y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=False)

  def predict(self, X_test):
    X = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    return self.model.predict(X)

