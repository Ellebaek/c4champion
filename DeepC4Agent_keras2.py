from Connect4Game import Connect4Game
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os


class DeepC4Agent:
    input_length = Connect4Game.ROW_COUNT * Connect4Game.COLUMN_COUNT * 3

    def __init__(self, name, load_models):
        self.name = name
        self.weights_file = "models/{0}_weights.h5".format(self.name)
        self.qnetwork = QNetwork(input_length=self.input_length)
        if load_models:
            self.load_weights_from_files()

    def load_weights_from_files(self):
        if os.path.exists(self.weights_file):
            print("Loading q-network weights for {0}".format(self.name))
            self.qnetwork.model.load_weights(self.weights_file)

    def save_model_weights(self):
        self.qnetwork.model.save_weights(self.weights_file)


class QNetwork:
    def __init__(self, input_length):
        self.model = Sequential()
        self.model.add(Dense(2048, input_dim=input_length, activation='relu', name='fc1', use_bias=False))
        self.model.add(Dense(Connect4Game.COLUMN_COUNT, name='fc2', use_bias=False))
        sgd = optimizers.SGD(lr=0.003)
        self.model.compile(loss='mse', optimizer=sgd, metrics=['mae'])
