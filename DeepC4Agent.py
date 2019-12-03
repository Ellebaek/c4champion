import tensorflow as tf
from Connect4Game import Connect4Game
import tensorflow.contrib.slim as slim


class DeepC4Agent():
    input_length = Connect4Game.ROW_COUNT * Connect4Game.COLUMN_COUNT

    def __init__(self, name):
        self.name = name
        # These lines establish the feed-forward part of the network used to choose actions
        self.inputs = tf.compat.v1.placeholder(shape=[None, self.input_length], dtype=tf.float32)
        self.keep_pct = tf.compat.v1.placeholder(shape=None, dtype=tf.float32)

        hidden = slim.fully_connected(self.inputs, 512,
                                      activation_fn=tf.nn.tanh,
                                      biases_initializer=None)

        hidden = slim.dropout(hidden, self.keep_pct)

        hidden = slim.fully_connected(hidden, 512,
                                      activation_fn=tf.nn.tanh,
                                      biases_initializer=None)

        hidden = slim.dropout(hidden, self.keep_pct)

        hidden = slim.fully_connected(hidden, 128,
                                      activation_fn=tf.nn.tanh,
                                      biases_initializer=None)

        hidden = slim.dropout(hidden, self.keep_pct)

        self.Qout = slim.fully_connected(hidden, Connect4Game.COLUMN_COUNT,
                                         activation_fn=None,
                                         biases_initializer=None)


        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        # self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        #self.actions_onehot = tf.one_hot(self.actions, Connect4Game.COLUMN_COUNT, dtype=tf.float32)

        # self.Q = tf.reduce_sum(tf.multiply(self.Q_out, self.actions_onehot), reduction_indices=1)

        self.nextQ = tf.compat.v1.placeholder(shape=[1, Connect4Game.COLUMN_COUNT], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        trainer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
        self.updateModel = trainer.minimize(loss)