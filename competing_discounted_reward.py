from math import log, exp
from qlearning_helper import open_actions, get_state, get_reward, board_size
from qlearning_helper import dup_mirror_input, discount_rewards, save_game, best_allowed_action, rand_index_filter
from DeepC4AgentTF import DeepC4AgentTF
from Connect4Game import Connect4Game
import numpy as np
import tensorflow as tf
import copy


ckpt_dir = 'checkpoints/'

def train_agent_against_list(sess, agent, opponents, episode_start_count=0):
    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    agents_turn_to_start = False
    for i in range(num_episodes):
        action_list = []
        reward_list = []
        # before adding episode_start_count training converges after approximately 60 generations
        e = e_init - (e_init - e_end) / e_steps * (i + 1 + episode_start_count)
        e2 = 0.20
        e3 = 0.05
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

        states = np.zeros((1, board_size*3))
        targetQ = np.zeros((1, Connect4Game.COLUMN_COUNT))
        # The game training
        while j < 100 and np.sum(open_actions(g)) > 0 and g.current_state == g.GAME_RUNNING:
            temp_states = states
            states = np.zeros((len(action_list)+1, board_size*3))
            temp_targetQ = targetQ
            targetQ = np.zeros((len(action_list)+1, Connect4Game.COLUMN_COUNT))
            if len(action_list) > 0:
                states[:-1, :] = temp_states
                targetQ[:-1, :] = temp_targetQ
            original_a = -1

            # agent
            if np.sum(open_actions(g)) > 0 and g.current_state == g.GAME_RUNNING:
                j += 1
                filter = open_actions(g)
                s = get_state(g)
                states[len(action_list), :] = s
                # Choose an action
                allQ = sess.run(agent.Qout, feed_dict={agent.inputs: s, agent.keep_pct: 1})
                if np.random.rand(1) < e:
                    # full random
                    a = rand_index_filter(filter)
                    original_a = a
                else:  # greedy
                    a = best_allowed_action(allQ, filter, 1)
                    original_a = np.argmax(allQ, axis=1)

                action_list.append(a)

                # initialize target
                targetQ[len(action_list)-1, :] = allQ

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
                #maxQ1 = get_max_future_reward_previous_player(g)

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
                #maxQ1 = get_max_future_reward_current_player(g)

            # update after opponent have played
            if original_a != a:
                targetQ[len(action_list)-1, original_a] = np.min(targetQ[len(action_list)-1, :]) # this improved game understanding a lot
            reward_list.append(r)
            #targetQ[0, a] = r + y * maxQ1
            rAll += r

        dr = discount_rewards(reward_list)
        for a_id in range(len(action_list)):
            targetQ[a_id, action_list[a_id]] = dr[a_id]

        states, targetQ = dup_mirror_input(states, targetQ)
        # Train our network using target and predicted Q values after each game with discounted reward
        _ = sess.run(agent.updateModel, feed_dict={agent.inputs: states, agent.keep_pct: 1, agent.nextQ: targetQ})

        jList.append(j)
        rList.append(sum(dr)/j)
        if (i + 1) % 50 == 0:
            sess.run(agent.training_episodes.assign_add(50))
            #ep = sess.run(agent.training_episodes, feed_dict={})
            #_ = sess.run(agent.training_episodes, feed_dict={agent.training_episodes: ep + 5})
            print("Training {0}   Episodes: {1} E: {2:.3f} J: {3:.3f} R: {4:.3f}".format(agent.name, i+1 + episode_start_count, e, np.mean(jList), np.mean(rList)))
            jList = []
            rList = []


def compete_and_return_score_list(sess, agent1, agent2, num_games):
    e2 = 0.20
    e3 = 0.05
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

        if g.current_state == Connect4Game.GAME_OVER and g.winner is not None:
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

        if g.current_state == Connect4Game.GAME_OVER and g.winner is not None:
            if agent1s_turn:
                wList.append(-1)
            else:
                wList.append(1)
        else:
            wList.append(0)

        # save game
        save_game(bList, "c4games/{0}_game{1}.txt".format(duelname, i+1))

    return wList








def get_next_challenger_id(current_id):
    idx = current_id + 1
    if idx >= num_agents:
        idx = 0
    return idx






num_agents = 6
num_generations = 21 #100
num_episodes = 100
max_num_episodes = 500 #4000
trial_length = 100
#y = .5
e_init = 1
e_end = 0.005
e_steps = max_num_episodes
final_challenger_id = -1

tf.compat.v1.reset_default_graph()

challenger_list = []
for i in range(num_agents + 1):
    challenger_list.append(DeepC4AgentTF("Agent{0}".format(i)))

init = tf.compat.v1.global_variables_initializer()
saver = tf.train.Saver()

with tf.compat.v1.Session() as sess:
    # TODO: conditional start from checkpoint or init
    sess.run(init)
    #saver.restore(sess, "{0}dr_test1_4.ckpt".format(ckpt_dir))

    CHAMPION = challenger_list[0]

    # train first opponent: Agent0
    # TODO: uncomment when re-training
    train_agent_against_list(sess, CHAMPION, [])
    e_init = 0.3
    # train with competition
    challenger_id = 0
    for i in range(1, num_generations + 1, 1):
        # "re-using" or continued learning have great impact on performance
        challenger_id = get_next_challenger_id(challenger_id)
        CHALLENGER = challenger_list[challenger_id]
        opponent_list = []
        score = -1
        episode_count = 0
        for j in range(min(num_agents, i)):
            if j != challenger_id:
                opponent_list.append(challenger_list[j])

        while score < 10 and episode_count < max_num_episodes:
            train_agent_against_list(sess, CHALLENGER,  opponent_list, episode_count)
            episode_count = episode_count + num_episodes
            win_list = compete_and_return_score_list(sess, CHAMPION, CHALLENGER, trial_length)
            score = np.sum(win_list)
            draws = np.size(np.where(np.array(win_list) == 0))
            print("Generation {0}: {1} against {2}. Score {3}. Draws: {4}".format(i,
                                                                                 CHAMPION.name,
                                                                                 CHALLENGER.name,
                                                                                 score,
                                                                                 draws))
        if score > 0:
            final_challenger_id = max(challenger_id - 1, 0)
            CHAMPION = CHALLENGER

    # let best models compete and save games
    duel_result = duel_and_save_games(sess, challenger_list[final_challenger_id], CHAMPION, "duel_{0}x{1}".format(num_generations, num_episodes))
    print("DUEL: {0} against CHAMPION ({1}). Result: {2}".format(challenger_list[final_challenger_id].name, CHAMPION.name, duel_result))


    #TODO: conditional save checkpoint
    # save session
    save_path = saver.save(sess, "{0}mir_test1_0.ckpt".format(ckpt_dir))
    #print("Session saved: {0}".format(save_path))
    print("Champion model: {0}".format(CHAMPION.name))