from math import log, exp
from qlearning_helper import open_actions, get_state, get_reward, get_max_future_reward_previous_player
from qlearning_helper import get_max_future_reward_current_player
from SimpleC4Agent import SimpleC4Agent
from DeepC4Agent import DeepC4Agent
from Connect4Game import Connect4Game
import numpy as np
import tensorflow as tf
import copy

def save_game(board_list, filename):
    f = open(filename, "w+")
    for board in board_list:
        f.write("{0}\n".format(board))
    f.close()

def best_allowed_action(q_values, open_actions, top):
    sorted_args = np.argsort(-q_values, 1)
    # multiple best
    a_bestv = q_values[0, sorted_args[0, 0]]
    filter2 = np.logical_and(q_values.flatten() == a_bestv, open_actions)
    if np.sum(filter2) > 0 and top == 1:
        a_best = rand_index_filter(filter2)
    else:
        count_down = max(top, np.sum(open_actions))
        # if none best are in filter, select the best remaining in filter
        for idx in range(Connect4Game.COLUMN_COUNT):
            if open_actions[sorted_args[0, idx]] == 1:
                a_best = sorted_args[0, idx]
                count_down = count_down - 1
                if count_down == 0:
                    break
    return a_best


def train_agent(sess, agent, opponent):
    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    agents_turn_to_start = False
    for i in range(num_episodes):
        e = e_init * 1. / log(i / 10 + exp(1))
        e2 = 0.30
        e3 = 0.10
        # Reset environment and get first new observation
        g = Connect4Game(announce_winner=True)
        rAll = 0
        j = 0
        agents_turn_to_start = not agents_turn_to_start
        # agents_turn = not agents_turn_to_start
        if opponent is not None and not agents_turn_to_start:
            # let opponent play first move
            filter = open_actions(g)
            s = get_state(g)
            allQopp = sess.run(opponent.Qout, feed_dict={opponent.inputs: s, opponent.keep_pct: 1})
            top = 3
            # introduce some random behaviour, deterministic player is too easy to learn to beat
            randval = np.random.rand(1)
            if randval > e2:
                top = 1
            elif randval > e3:
                top = 2
            a = best_allowed_action(allQopp, filter, top)
            g.play_piece(g.first_empty_row(a), a)

        # The game training
        while j < 100 and np.sum(open_actions(g)) > 0 and g.current_state == g.GAME_RUNNING:

            targetQ = np.zeros((1, Connect4Game.COLUMN_COUNT))

            # agent
            if np.sum(open_actions(g)) > 0 and g.current_state == g.GAME_RUNNING:
                j += 1
                filter = open_actions(g)
                s = get_state(g)
                # Choose an action
                allQ = sess.run(agent.Qout, feed_dict={agent.inputs: s, agent.keep_pct: 1})
                if np.random.rand(1) < e:
                    # full random
                    a = rand_index_filter(filter)
                else:  # greedy
                    a = best_allowed_action(allQ, filter, 1)

                # initialize target
                targetQ = allQ

                # Get new state and reward from environment
                #if g.first_empty_row(a) < 0:
                    # continue
                #    targetQ[0, a] = 0  # penalty for trying to play outside board
                #    r = 0
                #else:
                g.play_piece(g.first_empty_row(a), a)
                # s1 = get_state(g)
                r = get_reward(g)

                # set expectations in case opponent is not allowed to play or do not exist
                maxQ1 = get_max_future_reward_previous_player(g)

            # then opponent
            if opponent is not None and np.sum(open_actions(g)) > 0 and g.current_state == g.GAME_RUNNING:
                j += 1
                filter = open_actions(g)
                op_s = get_state(g)
                allQopp = sess.run(opponent.Qout, feed_dict={opponent.inputs: op_s, opponent.keep_pct: 1})
                top = 3
                # introduce some random behaviour, deterministic player is too easy to learn to beat
                randval = np.random.rand(1)
                if randval > e2:
                    top = 1
                elif randval > e3:
                    top = 2
                op_a = best_allowed_action(allQopp, filter, top)
                g.play_piece(g.first_empty_row(op_a), op_a)
                op_rew = get_reward(g)

                r = r - 0.9*op_rew

                # update expectations in case opponent was allowed to move
                maxQ1 = get_max_future_reward_current_player(g)

            # update after opponent have played
            targetQ[0, a] = r + y * maxQ1

            # Train our network using target and predicted Q values
            # Changed from s1 to s
            _ = sess.run(agent.updateModel, feed_dict={agent.inputs: s, agent.keep_pct: 1, agent.nextQ: targetQ})
            rAll += r

        jList.append(j)
        rList.append(rAll)
        if (i + 1) % 100 == 0:
            print("Training {0}   Episodes: {1} E: {2:.3f} J: {3:.3f} R: {4:.3f}".format(agent.name, i+1, e, np.mean(jList), np.mean(rList)))
            jList = []
            rList = []


def train_agent_against_list(sess, agent, opponents):
    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    agents_turn_to_start = False
    for i in range(num_episodes):
        e = e_init * 1. / log(i / 10 + exp(1))
        e2 = 0.30
        e3 = 0.10
        # Reset environment and get first new observation
        g = Connect4Game(announce_winner=True)
        rAll = 0
        j = 0
        op_idx = -1
        if len(opponents) > 0:
            op_idx = np.random.randint(len(opponents))

        agents_turn_to_start = not agents_turn_to_start
        # agents_turn = not agents_turn_to_start
        if op_idx > -1 and not agents_turn_to_start:
            opponent = opponents[op_idx]
            # let opponent play first move
            filter = open_actions(g)
            s = get_state(g)
            allQopp = sess.run(opponent.Qout, feed_dict={opponent.inputs: s, opponent.keep_pct: 1})
            top = 3
            # introduce some random behaviour, deterministic player is too easy to learn to beat
            randval = np.random.rand(1)
            if randval > e2:
                top = 1
            elif randval > e3:
                top = 2
            a = best_allowed_action(allQopp, filter, top)
            g.play_piece(g.first_empty_row(a), a)

        # The game training
        while j < 100 and np.sum(open_actions(g)) > 0 and g.current_state == g.GAME_RUNNING:

            targetQ = np.zeros((1, Connect4Game.COLUMN_COUNT))

            # agent
            if np.sum(open_actions(g)) > 0 and g.current_state == g.GAME_RUNNING:
                j += 1
                filter = open_actions(g)
                s = get_state(g)
                # Choose an action
                allQ = sess.run(agent.Qout, feed_dict={agent.inputs: s, agent.keep_pct: 1})
                if np.random.rand(1) < e:
                    # full random
                    a = rand_index_filter(filter)
                else:  # greedy
                    a = best_allowed_action(allQ, filter, 1)

                # initialize target
                targetQ = allQ

                # Get new state and reward from environment
                #if g.first_empty_row(a) < 0:
                    # continue
                #    targetQ[0, a] = 0  # penalty for trying to play outside board
                #    r = 0
                #else:
                g.play_piece(g.first_empty_row(a), a)
                # s1 = get_state(g)
                r = get_reward(g)

                # set expectations in case opponent is not allowed to play or do not exist
                maxQ1 = get_max_future_reward_previous_player(g)

            # then opponent
            if op_idx > -1 and np.sum(open_actions(g)) > 0 and g.current_state == g.GAME_RUNNING:
                opponent = opponents[op_idx]
                j += 1

                filter = open_actions(g)
                op_s = get_state(g)
                allQopp = sess.run(opponent.Qout, feed_dict={opponent.inputs: op_s, opponent.keep_pct: 1})
                top = 3
                # introduce some random behaviour, deterministic player is too easy to learn to beat
                randval = np.random.rand(1)
                if randval > e2:
                    top = 1
                elif randval > e3:
                    top = 2
                op_a = best_allowed_action(allQopp, filter, top)
                g.play_piece(g.first_empty_row(op_a), op_a)
                op_rew = get_reward(g)

                r = r - 0.9*op_rew

                # update expectations in case opponent was allowed to move
                maxQ1 = get_max_future_reward_current_player(g)

            # update after opponent have played
            targetQ[0, a] = r + y * maxQ1

            # Train our network using target and predicted Q values
            # Changed from s1 to s
            _ = sess.run(agent.updateModel, feed_dict={agent.inputs: s, agent.keep_pct: 1, agent.nextQ: targetQ})
            rAll += r

        jList.append(j)
        rList.append(rAll)
        if (i + 1) % 100 == 0:
            print("Training {0}   Episodes: {1} E: {2:.3f} J: {3:.3f} R: {4:.3f}".format(agent.name, i+1, e, np.mean(jList), np.mean(rList)))
            jList = []
            rList = []


def compete_and_return_score_list(sess, agent1, agent2, num_games):
    e2 = 0.30
    e3 = 0.10
    agent1_to_start = False
    wList = []
    for i in range(num_games):
        d = False
        j = 0
        g = Connect4Game(announce_winner=True)
        agent1_to_start = not agent1_to_start
        agent1s_turn = not agent1_to_start
        while j < 50 and np.sum(open_actions(g)) > 0 and not d:
            agent1s_turn = not agent1s_turn
            j += 1
            s = get_state(g)
            filter = open_actions(g)
            top = 3
            # introduce some random behaviour otherwise two games will settle it
            randval = np.random.rand(1)
            if randval > e2:
                top = 1
            elif randval > e3:
                top = 2
                # a = rand_index_filter(filter)
            if agent1s_turn:
                allQ = sess.run(agent1.Qout, feed_dict={agent1.inputs: s, agent1.keep_pct: 1})
                a = best_allowed_action(allQ, filter, top)
            else:
                allQ = sess.run(agent2.Qout, feed_dict={agent2.inputs: s, agent2.keep_pct: 1})
                a = best_allowed_action(allQ, filter, top)

            g.play_piece(g.first_empty_row(a), a)
            d = g.current_state == Connect4Game.GAME_OVER

        if g.current_state == Connect4Game.GAME_OVER:
            if agent1s_turn:
                wList.append(-1)
            else:
                wList.append(1)
        else:
            wList.append(0)

    return wList

def duel_and_save_games(sess, agent1, agent2, duelname):
    agent1_to_start = False
    wList = []
    for i in range(2):
        bList = []
        d = False
        j = 0
        g = Connect4Game(announce_winner=True)
        agent1_to_start = not agent1_to_start
        agent1s_turn = not agent1_to_start
        while j < 50 and np.sum(open_actions(g)) > 0 and not d:
            agent1s_turn = not agent1s_turn
            j += 1
            s = get_state(g)
            filter = open_actions(g)
            if agent1s_turn:
                allQ = sess.run(agent1.Qout, feed_dict={agent1.inputs: s, agent1.keep_pct: 1})
                a = best_allowed_action(allQ, filter, 1)
            else:
                allQ = sess.run(agent2.Qout, feed_dict={agent2.inputs: s, agent2.keep_pct: 1})
                a = best_allowed_action(allQ, filter, 1)

            g.play_piece(g.first_empty_row(a), a)
            d = g.current_state == Connect4Game.GAME_OVER
            bList.append(copy.deepcopy(g.board_position))

        if g.current_state == Connect4Game.GAME_OVER:
            if agent1s_turn:
                wList.append(-1)
            else:
                wList.append(1)
        else:
            wList.append(0)

        # save game
        save_game(bList, "c4games/{0}_game{1}.txt".format(duelname, i+1))

    return wList



def rand_index_filter(filter):
    f_idx = np.random.randint(np.sum(filter))
    return np.where(filter == 1)[0][f_idx]













num_iterations = 15
num_episodes = 1002
trial_length = 100
y = .5
e_init = 1
final_challenger_id = 6

tf.compat.v1.reset_default_graph()

challenger_list = []
for i in range(num_iterations + 1):
    challenger_list.append(DeepC4Agent("Agent{0}".format(i)))

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    CHAMPION = challenger_list[0]

    # train first opponent: Agent0
    train_agent_against_list(sess, CHAMPION, [])
    e_init = 0.5
    # train with competition
    for i in range(1, num_iterations + 1, 1):
        opponent_list = []
        CHALLENGER = challenger_list[i]
        for j in range(i):
            opponent_list.append(challenger_list[i - j - 1])
            if len(opponent_list) == 5:
                break

        train_agent_against_list(sess, CHALLENGER,  opponent_list)
        win_list = compete_and_return_score_list(sess, CHAMPION, CHALLENGER, trial_length)
        score = np.sum(win_list)
        draws = np.size(np.where(np.array(win_list) == 0))
        print("Iteration {0}: {1} against {2}. Score {3}. Draws: {4}".format(i,
                                                                             CHAMPION.name,
                                                                             CHALLENGER.name,
                                                                             score,
                                                                             draws))
        if score > 0:
            CHAMPION = CHALLENGER

    # let selected models compete and save games
    duel_result = duel_and_save_games(sess, challenger_list[final_challenger_id], CHAMPION, "duel_{0}x{1}".format(num_iterations,num_episodes))
    print("DUEL: {0} against CHAMPION ({1}). Result: {2}".format(challenger_list[final_challenger_id].name, CHAMPION.name, duel_result))
    #    f = open(filename, "w+")
    #    for board in bList:
    #        f.write("{0}\n".format(board))
    #    f.close()
    #    return g, bList, allQList


#TODO: migrate to keras
#TODO: Save champion model and maybe random challenger model to disk
#TODO: Read saved models from disk and make them compete in explorer window

#  g, bl, al = play_and_save_game(sess, "c4games/ex1.txt")