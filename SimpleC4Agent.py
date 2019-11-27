import tensorflow as tf
from Connect4Game import Connect4Game


class SimpleC4Agent():
    input_length = Connect4Game.ROW_COUNT * Connect4Game.COLUMN_COUNT

    def __init__(self, name):
        self.name = name
        self.inputs1 = tf.placeholder(shape=[1, self.input_length], dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform([self.input_length, Connect4Game.COLUMN_COUNT], 0.001, 0.01))
        self.Qout = tf.matmul(self.inputs1, self.W)
        #self.pred_sort = tf.argsort(self.Qout, 1, direction='DESCENDING')

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[1, Connect4Game.COLUMN_COUNT], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        self.trainer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
        self.updateModel = self.trainer.minimize(self.loss)
