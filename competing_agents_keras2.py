# import keras.backend as K
# import matplotlib.pyplot as plt
# import scipy.misc
from __future__ import division
from qlearning_helper import open_actions, get_state, get_reward
from qlearning_helper import dup_mirror_input, discount_rewards, best_allowed_action, rand_index_filter
from DeepC4Agent_keras import DeepC4Agent
from Connect4Game import Connect4Game
from ExperienceBuffer import ExperienceBuffer
import numpy as np


def train_agent_against_list(agent, opponents, num_games, episode_start_count=0):
    # create lists to contain total rewards and steps per episode
    episode_len_list = []
    r_list = []
    loss_list = []
    mae_list = []
    agents_turn_to_start = False
    for episode_no in range(num_games):
        action_list = []
        opp_action_list = []
        reward_list = []
        state_list = []
        target_q_list = []

        e = e_init - (e_init - e_end) / annealing_steps * (episode_no + 1 + episode_start_count)
        e2 = 0.20
        e3 = 0.05
        # Reset environment and get first new observation
        g = Connect4Game(announce_winner=True)
        episode_len = 0
        opp_idx = -1
        if len(opponents) > 0:
            opp_idx = np.random.randint(len(opponents))

        agents_turn_to_start = not agents_turn_to_start
        if opp_idx > -1 and not agents_turn_to_start:
            opponent = opponents[opp_idx]
            # let opponent play first move
            s = get_state(g)
            all_q_opp = opponent.qnetwork.model.predict(s)
            top = 3
            # introduce some random behaviour, deterministic player is too easy to learn to beat
            rand_val = np.random.rand(1)
            if rand_val > e2:
                top = 1
            elif rand_val > e3:
                top = 2
            opp_a = best_allowed_action(all_q_opp, open_actions(g), top)
            opp_action_list.append(opp_a)
            episode_len += 1
            g.play_piece(g.first_empty_row(opp_a), opp_a)

        # The game training
        while episode_len < max_num_step and np.sum(open_actions(g)) > 0 and g.current_state == g.GAME_RUNNING:
            r = 0
            # agent
            if np.sum(open_actions(g)) > 0 and g.current_state == g.GAME_RUNNING:
                episode_len += 1
                s = get_state(g)
                state_list.append(s)
                # Choose an action
                all_q = agent.qnetwork.model.predict(s)
                if np.random.rand(1) < e:
                    # full random
                    a = rand_index_filter(open_actions(g))
                    original_a = a
                else:  # greedy
                    a = best_allowed_action(all_q, open_actions(g), 1)
                    original_a = np.argmax(all_q, axis=1)

                action_list.append(a)

                # initialize target
                target_q_list.append(all_q)
                # update target if non-open action was originally preferred
                if original_a != a:
                    # this update improved game understanding a lot
                    target_q_list[-1][0][original_a] = np.min(target_q_list[-1])

                g.play_piece(g.first_empty_row(a), a)
                # initiate reward
                r = get_reward(g)

            # then opponent
            if opp_idx > -1 and np.sum(open_actions(g)) > 0 and g.current_state == g.GAME_RUNNING:
                opponent = opponents[opp_idx]
                episode_len += 1

                opp_s = get_state(g)
                all_q_opp = opponent.qnetwork.model.predict(opp_s)
                top = 3
                # introduce some random behaviour, deterministic player is too easy to learn to beat
                rand_val = np.random.rand(1)
                if rand_val > e2:
                    top = 1
                elif rand_val > e3:
                    top = 2
                opp_a = best_allowed_action(all_q_opp, open_actions(g), top)
                opp_action_list.append(opp_a)
                g.play_piece(g.first_empty_row(opp_a), opp_a)
                opp_rew = get_reward(g)

                # update reward after opponent move
                r = r - 0.9 * opp_rew

            reward_list.append(r)

        # update entire target with discounted rewards once game is over
        dr = discount_rewards(reward_list, gamma=y)
        for a_id in range(len(action_list)):
            target_q_list[a_id][0][action_list[a_id]] = dr[a_id]

        if gen_count <= experience_run_in:
            train_states_ready, train_target_q_ready = dup_mirror_input(np.concatenate(state_list),
                                                                        np.concatenate(target_q_list))
            if gen_count > experience_run_in - pre_buffer:
                # build initial experience buffer
                shared_experience.add_from_lists(action_list, state_list, dr)
        else:
            shared_experience.add_from_lists(action_list, state_list, dr)

            train_batch = shared_experience.sample(batch_size)

            # Separate the batch into its components
            train_states = np.concatenate(train_batch[:, 0].tolist())
            train_actions = train_batch[:, 1]
            train_rewards = train_batch[:, 2]

            # obtain new refreshed targetQ's
            train_target_q = agent.qnetwork.model.predict(train_states)
            for a_id in range(len(train_actions)):
                train_target_q[a_id, train_actions[a_id]] = train_rewards[a_id]

            train_states_ready, train_target_q_ready = dup_mirror_input(
                np.vstack([train_states, np.concatenate(state_list)]),
                np.vstack([train_target_q, np.concatenate(target_q_list)]))

        # train network using target and predicted Q values after each game with discounted reward
        loss, mae = agent.qnetwork.model.train_on_batch(train_states_ready, train_target_q_ready)

        loss_list.append(loss)
        mae_list.append(mae)
        episode_len_list.append(episode_len)
        r_list.append(sum(dr) / episode_len)
        if (episode_start_count + episode_no + 1) % print_interval == 0:
            s = "Training {0}   Episodes: {1} E: {2:.3f} L: {3:.3f} R: {4:.3f} Loss:{5:.3f} MAE:{6:.3f} Buffer:{7}"
            print(s.format(agent.name,
                           episode_no + 1 + episode_start_count,
                           e,
                           np.mean(episode_len_list),
                           np.mean(r_list),
                           np.mean(loss),
                           np.mean(mae),
                           len(shared_experience.buffer)))


def compete_and_return_score_list(agent1, agent2, num_games):
    e2 = 0.20
    e3 = 0.05
    agent1_to_start = False
    w_list = []
    for episode_no in range(num_games):
        d = False
        episode_len = 0
        g = Connect4Game(announce_winner=True)
        agent1_to_start = not agent1_to_start
        agent1s_turn = not agent1_to_start
        while episode_len < max_num_step and np.sum(open_actions(g)) > 0 and not d:
            agent1s_turn = not agent1s_turn
            episode_len += 1
            s = get_state(g)
            action_filter = open_actions(g)
            top = 3
            # introduce some random behaviour otherwise two games will settle it
            rand_val = np.random.rand(1)
            if rand_val > e2:
                top = 1
            elif rand_val > e3:
                top = 2
                # a = rand_index_filter(filter)
            if agent1s_turn:
                all_q = agent1.qnetwork.model.predict(s)
                a = best_allowed_action(all_q, action_filter, top)
            else:
                all_q = agent2.qnetwork.model.predict(s)
                a = best_allowed_action(all_q, action_filter, top)

            g.play_piece(g.first_empty_row(a), a)
            d = g.current_state == Connect4Game.GAME_OVER

        if g.current_state == Connect4Game.GAME_OVER and g.winner is not None:
            if agent1s_turn:
                w_list.append(-1)
            else:
                w_list.append(1)
        else:
            w_list.append(0)

    return w_list


batch_size = 64  # How many experiences to add on top of played episode in training step
y = 0.85  # Discount factor gamma = 0.95
load_models = True  # Whether to load saved models or not
experience_run_in = 20  # Number of generations to train before starting to sampling from experience
pre_buffer = 2  # Number of generations to have ready in experience before sampling (pre_buffer << experience_run_in)
print_interval = 50  # How often to print status
compete_interval = 250  # number of training episodes between competing with champion model
trial_length = 50  # how many games to settle if champion can be beaten
score_margin = 5  # margin number of wins to convincing win against champion
max_num_episodes = 2000  # max number of episodes in generation
e_init = 0.3  # Starting chance of random action
e_end = 0.005  # Ending chance of random action
annealing_steps = max_num_episodes  # Steps of training to reduce from start_e -> end_e
max_num_step = 50  # Maximum allowed episode length
num_agents = 6  # number of agents live and training with each other
num_generations = 10000  # how long to run the program


def get_next_challenger_id(current_id):
    idx = current_id + 1
    if idx >= num_agents:
        idx = 0
    return idx


# Setup our Q-networks
challenger_list = []
for i in range(num_agents + 1):
    challenger_list.append(DeepC4Agent(name="AgentK{0}".format(i), load_models=load_models))

shared_experience = ExperienceBuffer(buffer_size=10000)
CHAMPION = challenger_list[0]

# train with competition
challenger_id = 0
for gen_count in range(1, num_generations + 1, 1):
    # "re-using" or continued learning have great impact on performance
    challenger_id = get_next_challenger_id(challenger_id)
    CHALLENGER = challenger_list[challenger_id]
    opponent_list = []
    score = -1
    episode_count = 0
    for j in range(num_agents):  # range(min(num_agents, i)):
        if j != challenger_id:
            opponent_list.append(challenger_list[j])

    while score < score_margin and episode_count < max_num_episodes:
        train_agent_against_list(CHALLENGER, opponent_list, compete_interval, episode_count)
        episode_count = episode_count + compete_interval
        win_list = compete_and_return_score_list(CHAMPION, CHALLENGER, trial_length)
        score = np.sum(win_list)
        draws = np.size(np.where(np.array(win_list) == 0))
        print("Generation {0}: {1} against {2}. Score {3}. Draws: {4}".format(gen_count,
                                                                              CHAMPION.name,
                                                                              CHALLENGER.name,
                                                                              score,
                                                                              draws))
    CHALLENGER.save_model_weights()
    # final_challenger_id = max(challenger_id - 1, 0)
    CHAMPION = CHALLENGER

print("Champion model: {0}".format(CHAMPION.name))

# let best models compete and save games
# duel_result = duel_and_save_games(sess,
# challenger_list[final_challenger_id],
# CHAMPION,
# "duel_{0}x{1}".format(num_generations, num_episodes))
# print("DUEL: {0} against CHAMPION ({1}).
# Result: {2}".format(challenger_list[final_challenger_id].name, CHAMPION.name, duel_result))
