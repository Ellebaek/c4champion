from Connect4Game import Connect4Game
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
import numpy as np
import os

class DeepC4Agent():
    input_length = Connect4Game.ROW_COUNT * Connect4Game.COLUMN_COUNT * 3

    def __init__(self, name, load_models):
        self.name = name
        self.main_weights_file = "models/{0}_main_weights.h5".format(self.name)
        self.target_weights_file = "models/{0}_target_weights.h5".format(self.name)
        self.main_network = QNetwork(input_length = self.input_length)
        self.target_network = QNetwork(input_length = self.input_length)
        # Make the networks equal
        self.update_target_network(1)
        if load_models:
            self.load_weights_from_files()

    def update_target_network(self, tau):
        updated_weights = (np.array(self.main_network.model.get_weights()) * tau) + (np.array(self.target_network.model.get_weights()) * (1 - tau))
        self.target_network.model.set_weights(updated_weights)

    def load_weights_from_files(self):
        if os.path.exists(self.main_weights_file):
            print("Loading main weights")
            self.main_network.model.load_weights(self.main_weights_file)
        if os.path.exists(self.target_weights_file):
            print("Loading target weights")
            self.target_network.model.load_weights(self.target_weights_file)

    def save_model_weights(self):
        self.target_network.model.save_weights(self.target_weights_file)
        self.main_network.model.save_weights(self.main_weights_file)

class QNetwork():
    def __init__(self, input_length):
        self.model = Sequential()
        self.model.add(Dense(512, input_dim=input_length, activation='relu', name='fc1', use_bias=False))
        self.model.add(Dense(2048, activation='relu', name='fc2', use_bias=False))
        self.model.add(Dense(1024, activation='relu', name='fc3', use_bias=False))
        self.model.add(Dense(128, activation='relu', name='fc4', use_bias=False))
        self.model.add(Dense(Connect4Game.COLUMN_COUNT, name='fc5', use_bias=False))
        sgd = optimizers.SGD(lr=0.001)
        self.model.compile(loss='mse', optimizer=sgd, metrics=['mae'])