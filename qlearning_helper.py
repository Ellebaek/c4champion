import numpy as np
from Connect4Game import Connect4Game

input_length = Connect4Game.ROW_COUNT * Connect4Game.COLUMN_COUNT


def open_actions(c4game):
    oa = [0 if c4game.first_empty_row(_x) < 0 else 1 for _x in range(c4game.COLUMN_COUNT)]
    return np.array(oa)

def get_state(c4game):
    board = np.array(c4game.board_position).reshape((1, input_length))
    # board = [[c4game.board_position[_r][_c] for _c in range(c4game.COLUMN_COUNT)] for _r in range(c4game.ROW_COUNT)]
    return board


def get_reward(c4game):
    rew = -0.01
    if c4game.current_state == Connect4Game.GAME_OVER:
        rew = 1
    return rew


def get_max_future_reward_previous_player(c4game):
    rew = 0
    # TODO: consider checking for only one spot left

    evaluation = c4game.evaluate_board()
    if c4game.current_state == c4game.GAME_RUNNING:
        if evaluation[c4game.previous_player * 2 - 1]:
            rew = 1
        else:
            rew = 0.5
    return rew
