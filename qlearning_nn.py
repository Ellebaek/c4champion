from math import log, exp
from qlearning_helper import open_actions, get_state, get_reward, get_max_future_reward_previous_player
from qlearning_helper import input_length

from Connect4Game import Connect4Game
import numpy as np
import tensorflow as tf
import copy


# import keras
# import matplotlib.pyplot as plt
# %matplotlib inline

# source
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0


# try a trained model
def play_and_save_game(sess, filename):
    g = Connect4Game(announce_winner=True)
    d = False
    j = 0
    bList = []
    allQList = []
    while j < 500 and not d and np.sum(open_actions(g)) > 0:
        j += 1
        s = get_state(g)
        # Choose an action using a sample from a dropout approximation of a bayesian q-network.
        a, allQ = sess.run([predict, Qout], feed_dict={inputs1: s, action_filter: np.diag(open_actions(g))})
        print(open_actions(g))
        print("{0}: {1} / {2}".format(j, a[0], g.first_empty_row(a[0])))
        g.play_piece(g.first_empty_row(a[0]), a[0])
        d = g.current_state == Connect4Game.GAME_OVER

        bList.append(copy.deepcopy(g.board_position))
        allQList.append(copy.deepcopy(allQ))

    f = open(filename, "w+")
    for board in bList:
        f.write("{0}\n".format(board))
    f.close()
    return g, bList, allQList


tf.reset_default_graph()
# These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1, input_length], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([input_length, Connect4Game.COLUMN_COUNT], 0.001, 0.01))
action_filter = tf.placeholder(shape=[Connect4Game.COLUMN_COUNT, Connect4Game.COLUMN_COUNT], dtype=tf.float32)
Qout = tf.matmul(inputs1, W)
Qout = tf.matmul(Qout, action_filter)
predict = tf.argmax(Qout, 1)

# Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1, Connect4Game.COLUMN_COUNT], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
updateModel = trainer.minimize(loss)

# Training the network
init = tf.global_variables_initializer()

# Set learning parameters
y = .3
e_init = 1
num_episodes = 50
# create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        e = e_init * 1. / log(i / 10 + exp(1))
        print("Episode: " + str(i) + " E: " + str(e))
        # Reset environment and get first new observation
        g = Connect4Game(announce_winner=True)
        rAll = 0
        d = False
        j = 0
        # The Q-Network
        while j < 100 and not d and np.sum(open_actions(g)) > 0:
            j += 1

            s = get_state(g)
            # Choose an action by greedily (with e chance of random action) from the Q-network
            a, allQ = sess.run([predict, Qout], feed_dict={inputs1: s, action_filter: np.diag(open_actions(g))})
            if np.random.rand(1) < e:
                a[0] = np.random.randint(Connect4Game.COLUMN_COUNT)
            # initialize target
            targetQ = allQ
            # if np.random.rand(1) < e:
            #    a[0] = env.action_space.sample()
            # Get new state and reward from environment
            if g.first_empty_row(a[0]) < 0:
                # continue
                targetQ[0, a[0]] = 0  # penalty for trying to play outside board
            else:
                g.play_piece(g.first_empty_row(a[0]), a[0])
                s1 = get_state(g)
                r = get_reward(g)
                d = g.current_state == Connect4Game.GAME_OVER

                # Obtain the Q' values by feeding the new state through our network
                # Q1 = sess.run(Qout,feed_dict={inputs1:s1})
                # Obtain maxQ' and set our target value for chosen action.
                # maxQ1 = np.max(Q1)
                maxQ1 = get_max_future_reward_previous_player(g)
                targetQ[0, a[0]] = r + y * maxQ1

            # Train our network using target and predicted Q values
            _, W1 = sess.run([updateModel, W], feed_dict={inputs1: s1, action_filter: np.identity(7), nextQ: targetQ})
            rAll += r
            # s = s1

        jList.append(j)
        rList.append(rAll)
    g, bl, al = play_and_save_game(sess, "c4games/ex1.txt")

# print("Percent of successful episodes: " + str(sum(rList)/num_episodes) + "%")
