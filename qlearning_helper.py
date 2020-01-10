import numpy as np
from Connect4Game import Connect4Game
import copy

board_size = Connect4Game.ROW_COUNT * Connect4Game.COLUMN_COUNT


def open_actions(c4game):
    oa = [0 if c4game.first_empty_row(_x) < 0 else 1 for _x in range(c4game.COLUMN_COUNT)]
    return np.array(oa)


def get_state42(c4game):
    board = np.array(c4game.board_position) + 0.01
    return board.reshape((1, board_size))


def get_state(c4game):
    boardP1 = (np.array(c4game.board_position) == -1).astype(int)
    boardP2 = (np.array(c4game.board_position) == 1).astype(int)
    boardEmpty = (np.array(c4game.board_position) == 0).astype(int)
    return np.concatenate((boardP1.reshape((1, board_size)), boardP2.reshape((1, board_size)), boardEmpty.reshape((1, board_size))), axis = 1)

gamma = 0.95
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def get_reward(c4game):
    rew = 0.01
    if c4game.current_state == Connect4Game.GAME_OVER:
        rew = 1
    return rew


def get_max_future_reward_previous_player(c4game):
    rew = 0
    # TODO: consider checking for only one spot left

    if c4game.current_state == c4game.GAME_RUNNING:
        evaluation = c4game.evaluate_board()
        if evaluation[c4game.previous_player * 2 - 1]:
            rew = 1
#            if not evaluation[c4game.current_player * 2 - 1]:
#                rew = 1
        else:
            rew = 0.5
    return rew


def get_max_future_reward_current_player(c4game):
    rew = 0
    # TODO: consider checking for only one spot left

    if c4game.current_state == c4game.GAME_RUNNING:
        evaluation = c4game.evaluate_board()
        if evaluation[c4game.current_player * 2 - 1]:
            rew = 1
            #if not evaluation[c4game.current_player * 2 - 1]:
            #    rew = 1
        else:
            rew = 0.5
    return rew


def save_game(board_list, filename):
    f = open(filename, "w+")
    for board in board_list:
        f.write("{0}\n".format(board))
    f.close()


def rand_index_filter(filter):
    f_idx = np.random.randint(np.sum(filter))
    return np.where(filter == 1)[0][f_idx]


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

def dup_mirror_input(states, targets):
    states_mirror = np.zeros((states.shape))
    for r in range(states.shape[0]):
        bP1Mirror = np.flip(states[r, 0:board_size].reshape((Connect4Game.ROW_COUNT, Connect4Game.COLUMN_COUNT)),1)
        bP2Mirror = np.flip(states[r, board_size:2*board_size].reshape((Connect4Game.ROW_COUNT, Connect4Game.COLUMN_COUNT)),1)
        bEmptyMirror = np.flip(states[r, 2*board_size:3*board_size].reshape((Connect4Game.ROW_COUNT, Connect4Game.COLUMN_COUNT)),1)
        states_mirror[r, :] = np.concatenate((bP1Mirror.reshape((1, board_size)),
                                              bP2Mirror.reshape((1, board_size)),
                                              bEmptyMirror.reshape((1, board_size))), axis=1)
    targets_mirror = np.flip(targets, axis=1)
    return np.concatenate((states, states_mirror), axis=0), np.concatenate((targets, targets_mirror), axis=0)