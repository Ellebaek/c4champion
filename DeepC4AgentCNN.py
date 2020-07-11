from Connect4Game import Connect4Game
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import numpy as np
import os


class DeepC4Agent:
    def __init__(self, name, load_models):
        self.name = name
        self.weights_file = "models/{0}_weights.h5".format(self.name)
        self.qnetwork = QNetworkCNN()
        if load_models:
            self.load_weights_from_files()

    def load_weights_from_files(self):
        if os.path.exists(self.weights_file):
            print("Loading q-network weights for {0}".format(self.name))
            self.qnetwork.model.load_weights(self.weights_file)

    def save_model_weights(self):
        self.qnetwork.model.save_weights(self.weights_file)


class QNetworkCNN:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size=3, activation='relu',
                              input_shape=(Connect4Game.ROW_COUNT, Connect4Game.COLUMN_COUNT, 1)))
        self.model.add(Conv2D(64, kernel_size=3, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(Connect4Game.COLUMN_COUNT, name='fc1', use_bias=False))
        sgd = optimizers.SGD(lr=0.003)
        self.model.compile(loss='mse', optimizer=sgd, metrics=['mae'])

    def predict(self, x):
        _x = x
        if _x.ndim == 2:
            _x = np.expand_dims(_x, axis=0)
        _x = np.expand_dims(_x, axis=3)
        return self.model.predict(_x)

    def train_on_batch(self, x, y):
        _x = np.expand_dims(x, axis=3)
        return self.model.train_on_batch(_x, y)
