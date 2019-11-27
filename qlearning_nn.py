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
    while j < 50 and not d and np.sum(open_actions(g)) > 0:
        j += 1
        s = get_state(g)

        a_sort, allQ = sess.run([pred_sort, Qout], feed_dict={inputs1: s})
        a = a_sort[0, 0]
        print(open_actions(g))
        print("{0}: {1} / {2}".format(j, a, g.first_empty_row(a)))
        g.play_piece(g.first_empty_row(a), a)
        d = g.current_state == Connect4Game.GAME_OVER

        bList.append(copy.deepcopy(g.board_position))
        allQList.append(copy.deepcopy(allQ))

    f = open(filename, "w+")
    for board in bList:
        f.write("{0}\n".format(board))
    f.close()
    return g, bList, allQList


def rand_index_filter(filter):
    f_idx = np.random.randint(np.sum(filter))
    return np.where(filter == 1)[0][f_idx]



tf.reset_default_graph()
# These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1, input_length], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([input_length, Connect4Game.COLUMN_COUNT], 0.001, 0.01))
# action_filter = tf.placeholder(shape=[Connect4Game.COLUMN_COUNT, Connect4Game.COLUMN_COUNT], dtype=tf.float32)
Qout = tf.matmul(inputs1, W)
# Qout2 = tf.matmul(Qout, action_filter)
pred_sort = tf.argsort(Qout, 1, direction='DESCENDING')
# predict = tf.argmax(Qout2, 1)

# Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1, Connect4Game.COLUMN_COUNT], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
updateModel = trainer.minimize(loss)

# Training the network
init = tf.global_variables_initializer()

# Set learning parameters
y = .5
e_init = 1
num_episodes = 1000
# create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        e = e_init * 1. / log(i / 10 + exp(1))
        if i % 100 == 0:
            print("Episode: " + str(i) + " E: " + str(e))
        # Reset environment and get first new observation
        g = Connect4Game(announce_winner=True)
        rAll = 0
        d = False
        j = 0
        # The Q-Network
        while j < 100 and not d and np.sum(open_actions(g)) > 0:
            j += 1

            filter = open_actions(g)
            s = get_state(g)
            # Choose an action by greedily (with e chance of random action) from the Q-network
            a_sort, allQ = sess.run([pred_sort, Qout], feed_dict={inputs1: s})
            # full random
            if np.random.rand(1) < e:
                a = rand_index_filter(filter)
            else: # greedy
                # multiple best
                a_bestv = allQ[0, a_sort[0, 0]]
                filter2 = np.logical_and(allQ.flatten() == a_bestv, filter)
                if np.sum(filter2) > 0:
                    a = rand_index_filter(filter2)
                else:
                    # if none best are in filter, select the best remaining in filter
                    for idx in range(Connect4Game.COLUMN_COUNT):
                        if filter[a_sort[0,idx]] == 1:
                            a = a_sort[0,idx]

            # initialize target
            targetQ = allQ
            # Get new state and reward from environment
            if g.first_empty_row(a) < 0:
                # continue
                targetQ[0, a] = 0  # penalty for trying to play outside board
                r = 0
            else:
                g.play_piece(g.first_empty_row(a), a)
                s1 = get_state(g)
                r = get_reward(g)
                d = g.current_state == Connect4Game.GAME_OVER

                # Obtain the Q' values by feeding the new state through our network
                # Q1 = sess.run(Qout,feed_dict={inputs1:s1})
                # Obtain maxQ' and set our target value for chosen action.
                # maxQ1 = np.max(Q1)
                maxQ1 = get_max_future_reward_previous_player(g)
                targetQ[0, a] = r + y * maxQ1

            # Train our network using target and predicted Q values
            _, W1 = sess.run([updateModel, W], feed_dict={inputs1: s1, nextQ: targetQ})
            rAll += r
            # s = s1

        jList.append(j)
        rList.append(rAll)
    g, bl, al = play_and_save_game(sess, "c4games/ex1.txt")

# print("Percent of successful episodes: " + str(sum(rList)/num_episodes) + "%")
