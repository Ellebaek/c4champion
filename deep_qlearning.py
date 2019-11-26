import tensorflow as tf
import numpy as np
import random
import copy

from Connect4Game import Connect4Game
from Q_Network import Q_Network
from qlearning_helper import open_actions, get_state, get_reward, get_max_future_reward_previous_player, input_length


class experience_buffer():
    def __init__(self, buffer_size=20):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:int(total_vars / 2)]):
        op_holder.append(tfVars[idx + int(total_vars / 2)].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + int(total_vars / 2)].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


# Set learning parameters
y = .3  # Discount factor.
num_episodes = 100  # Total number of episodes to train network for.
tau = 0.001  # Amount to update target network at each step.
batch_size = 8  # Size of training batch
startE = 1  # Starting chance of random action
endE = 0.1  # Final chance of random action
anneling_steps = 2000  # How many steps of training to reduce startE to endE.
pre_train_steps = 1000  # Number of steps used before training updates begin.

tf.reset_default_graph()

q_net = Q_Network()
target_net = Q_Network()

init = tf.compat.v1.global_variables_initializer()
trainables = tf.compat.v1.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)
myBuffer = experience_buffer()

# create lists to contain total rewards and steps per episode
jList = []
jMeans = []
rList = []
rMeans = []
bList = []
allQList = []
with tf.Session() as sess:
    sess.run(init)
    updateTarget(targetOps, sess)
    e = startE
    stepDrop = (startE - endE) / anneling_steps
    total_steps = 0

    for i in range(num_episodes):
        g = Connect4Game(announce_winner=True)
        rAll = 0
        d = False
        j = 0
        while j < 100 and not d and np.sum(open_actions(g)) > 0:
            j += 1
            s = get_state(g)
            # Choose an action using a sample from a dropout approximation of a bayesian q-network.
            a, allQ = sess.run([q_net.predict, q_net.Q_out],
                               feed_dict={q_net.inputs: s, q_net.keep_per: (1 - e) + 0.1})
            #a = a[0]

            # Get new state and reward from environment
            # initialize target
            targetQ = allQ
            # Get new state and reward from environment
            if g.first_empty_row(a[0]) < 0:
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

            myBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))

            if e > endE and total_steps > pre_train_steps:
                e -= stepDrop

            if total_steps > pre_train_steps and total_steps % 20 == 0:
                # We use Double-DQN training algorithm
                trainBatch = myBuffer.sample(batch_size)
                Q1 = sess.run(q_net.predict, feed_dict={q_net.inputs: np.vstack(trainBatch[:, 0]), q_net.keep_per: 1.0})
                Q2 = sess.run(target_net.Q_out,
                              feed_dict={target_net.inputs: np.vstack(trainBatch[:, 0]), target_net.keep_per: 1.0})
                end_multiplier = -(trainBatch[:, 4] - 1)
                doubleQ = Q2[range(batch_size), Q1]
                targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                _ = sess.run(q_net.updateModel,
                             feed_dict={q_net.inputs: np.vstack(trainBatch[:, 0]), q_net.nextQ: targetQ,
                                        q_net.keep_per: 1.0, q_net.actions: trainBatch[:, 1]})
                updateTarget(targetOps, sess)

            rAll += r
            if i + 1 == num_episodes:
                bList.append(copy.deepcopy(g.board_position))
                allQList.append(copy.deepcopy(allQ))
            #s = s1
            total_steps += 1

        jList.append(j)
        rList.append(rAll)
        if i % 10 == 0 and i != 0:
            r_mean = np.mean(rList[-10:])
            j_mean = np.mean(jList[-10:])
            print("Mean Reward: " + str(r_mean) + " Total Steps: " + str(total_steps) + " p: " + str(e))
            rMeans.append(r_mean)
            jMeans.append(j_mean)

print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")

f = open("c4games/ex2.txt", "w+")
for board in bList:
    f.write("{0}\n".format(board))
f.close()