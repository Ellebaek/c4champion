import tensorflow as tf
from Connect4Game import Connect4Game
import tensorflow.contrib.slim as slim

# TODO: implement 5x5 convolution  with stride 1x1 and no padding, minimum 50 filters
class DeepC4AgentTF():
    input_length = Connect4Game.ROW_COUNT * Connect4Game.COLUMN_COUNT * 3

    def __init__(self, name):
        self.name = name
        # These lines establish the feed-forward part of the network used to choose actions
        self.inputs = tf.compat.v1.placeholder(shape=[None, self.input_length], dtype=tf.float32)
        self.keep_pct = tf.compat.v1.placeholder(shape=None, dtype=tf.float32)
        self.training_episodes = tf.Variable(0.0)

        hidden1 = slim.fully_connected(self.inputs, 512,
                                       activation_fn=tf.nn.relu,
                                       biases_initializer=None,
                                       scope="{0}/fc1".format(name))

        dropout1 = slim.dropout(hidden1, self.keep_pct)

        hidden2 = slim.fully_connected(dropout1, 2048,
                                       activation_fn=tf.nn.relu,
                                       biases_initializer=None,
                                       scope="{0}/fc2".format(name))

        dropout2 = slim.dropout(hidden2, self.keep_pct)

        hidden3 = slim.fully_connected(dropout2, 1024,
                                       activation_fn=tf.nn.relu,
                                       biases_initializer=None,
                                       scope="{0}/fc3".format(name))

        dropout3 = slim.dropout(hidden3, self.keep_pct)

        hidden4 = slim.fully_connected(dropout3, 128,
                                       activation_fn=tf.nn.relu,
                                       biases_initializer=None,
                                       scope="{0}/fc4".format(name))

        dropout4 = slim.dropout(hidden4, self.keep_pct)

        self.Qout = slim.fully_connected(dropout4, Connect4Game.COLUMN_COUNT,
                                         activation_fn=None,
                                         biases_initializer=None,
                                         scope="{0}/fc5".format(name))


        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.compat.v1.placeholder(shape=[1, Connect4Game.COLUMN_COUNT], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        trainer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
        self.updateModel = trainer.minimize(loss)
