from Connect4Game import Connect4Game
import numpy as np
import tensorflow as tf
from DeepC4AgentTF import DeepC4AgentTF
from DeepC4Agent_keras import DeepC4Agent
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

ckpt_dir = 'checkpoints/'

num_layers = 5
CHAMPION = DeepC4AgentTF("Agent4")
saver = tf.train.Saver()
vars_global = tf.global_variables()

with tf.compat.v1.Session() as sess:
    #sess.run(init)
    saver.restore(sess, "{0}test2.ckpt".format(ckpt_dir))
    model_vars = {}
    for var in vars_global:
        try:
            model_vars[var.name] = var.eval()
            print(var.name)
            #Agent4/fc1/weights:0
            #Agent4/fc2/weights:0
            #Agent4/fc3/weights:0
            #Agent4/fc4/weights:0
            #Agent4/fc5/weights:0
        except:
            print("For var={}, an exception occurred".format(var.name))
    model = Sequential()
    model.add(Dense(512, input_dim=126, activation='relu', name='fc1', use_bias=False))
    model.add(Dense(Connect4Game.COLUMN_COUNT, name='output'))
    sgd = optimizers.SGD(lr=0.001)
    model.compile(loss='mse', optimizer=sgd, metrics=['mae'])

    #model.layers[0].set_weights([model_vars['Agent4/fc1/weights:0']])
    keras_champion = DeepC4Agent("champion")
    for i in range(len(keras_champion.model.layers)):
        keras_champion.model.layers[i].set_weights([model_vars["{0}/fc{1}/weights:0".format(CHAMPION.name, i+1)]])

    print(keras_champion.model.summary())

    # TODO: save keras model to disk
    # TODO: create new module to load and compete with keras model
