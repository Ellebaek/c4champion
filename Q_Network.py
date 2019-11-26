import tensorflow as tf
import tensorflow.contrib.slim as slim
from Connect4Game import Connect4Game


class Q_Network():
    input_length = Connect4Game.ROW_COUNT * Connect4Game.COLUMN_COUNT

    def __init__(self):
        # These lines establish the feed-forward part of the network used to choose actions
        self.inputs = tf.placeholder(shape=[None, self.input_length], dtype=tf.float32)
        self.Temp = tf.placeholder(shape=None, dtype=tf.float32)
        self.keep_per = tf.placeholder(shape=None, dtype=tf.float32)
        self.action_filter = tf.placeholder(shape=[Connect4Game.COLUMN_COUNT, Connect4Game.COLUMN_COUNT], dtype=tf.float32)

        hidden = slim.fully_connected(self.inputs, 32, activation_fn=tf.nn.tanh, biases_initializer=None)
        hidden = slim.dropout(hidden, self.keep_per)
        self.Q_out = slim.fully_connected(hidden, Connect4Game.COLUMN_COUNT, activation_fn=None,
                                          biases_initializer=None)
        self.Q_out = tf.matmul(self.Q_out, action_filter)
        self.predict = tf.argmax(self.Q_out, 1)
        self.Q_dist = tf.nn.softmax(self.Q_out / self.Temp)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, Connect4Game.COLUMN_COUNT, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Q_out, self.actions_onehot), reduction_indices=1)

        self.nextQ = tf.placeholder(shape=[None], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.nextQ - self.Q))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)
        self.updateModel = trainer.minimize(loss)
