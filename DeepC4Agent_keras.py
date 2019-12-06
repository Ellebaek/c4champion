from Connect4Game import Connect4Game
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

# TODO: implement 5x5 convolution  with stride 1x1 and no padding, minimum 50 filters
class DeepC4Agent():
    input_length = Connect4Game.ROW_COUNT * Connect4Game.COLUMN_COUNT * 3

    def __init__(self, name):
        self.name = name
        self.keep_pct = 1
        self.model = Sequential()
        self.model.add(Dense(512, input_dim=self.input_length, activation='relu', name='fc1'))
        self.model.add(Dropout(1 - self.keep_pct))
        self.model.add(Dense(2048, activation='relu', name='fc2'))
        self.model.add(Dropout(1 - self.keep_pct))
        self.model.add(Dense(1024, activation='relu', name='fc3'))
        self.model.add(Dropout(1 - self.keep_pct))
        self.model.add(Dense(128, activation='relu', name='fc4'))
        self.model.add(Dropout(1 - self.keep_pct))
        self.model.add(Dense(Connect4Game.COLUMN_COUNT, name='output'))
        sgd = optimizers.SGD(lr=0.001)
        self.model.compile(loss='mse', optimizer=sgd, metrics=['mae'])
