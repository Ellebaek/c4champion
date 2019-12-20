import numpy as np

class Connect4Game():
    ROW_COUNT = 6
    COLUMN_COUNT = 7

    # game "states"
    GAME_RUNNING = 1
    GAME_OVER = 2

    def __init__(self, announce_winner=False):
        self.current_player = 1
        self.previous_player = 2
        self.board_position = self.new_board()
        self.pieces_played = 0

        self.announce_winner = announce_winner
        self.current_state = self.GAME_RUNNING
        self.winner = None

    def new_board(self):
        # Create the 'empty' board of 0's
        board = [[0 for _x in range(self.COLUMN_COUNT)] for _y in range(self.ROW_COUNT)]
        return board

    #def get_winner(self):
    #    w = None
    #    if self.evaluate_board(self.previous_player) == 4:
    #        w = self.previous_player
    #    return w

    def evaluate_strip(self, pos1, pos2, pos3, pos4):
        pot1 = False
        pot2 = False
        sum1 = 0
        sum2 = 0
        a = np.array([pos1, pos2, pos3, pos4])
        s = np.sum(a)
        if s == 0:
            if pos1 == 0 and pos2 == 0 and pos3 == 0 and pos4 == 0:
                pot1 = True
                pot2 = True
        elif a.max() == 0 or s == -4:
            sum1 = s
            pot1 = True
        elif a.min() == 0 or s == 4:
            sum2 = s
            pot2 = True

        return [sum1, pot1, sum2, pot2]

    def evaluate_board(self):
        strip_eval = None
        center_height = self.ROW_COUNT - 1 # - self.first_empty_row(3)
        sum1 = 0
        sum2 = 0
        pot1 = False
        pot2 = False
        # check vertical
        for column in range(self.COLUMN_COUNT):
            height = self.ROW_COUNT - 1  # - self.first_empty_row(column)
            i = 0
            while i <= height - 3:
                row_id = self.ROW_COUNT - 1 - i
                strip_eval = self.evaluate_strip(self.board_position[row_id][column],
                                                 self.board_position[row_id-1][column],
                                                 self.board_position[row_id-2][column],
                                                 self.board_position[row_id-3][column])
                sum1 = min(sum1, strip_eval[0])
                pot1 = pot1 or strip_eval[1]
                sum2 = max(sum2, strip_eval[2])
                pot2 = pot2 or strip_eval[3]
                i = i + 1

        # check horizontal
        for row in range(center_height):
            row_id = self.ROW_COUNT - 1 - row
            i = 0
            while i < 4:
                strip_eval = self.evaluate_strip(self.board_position[row_id][i],
                                                 self.board_position[row_id][i+1],
                                                 self.board_position[row_id][i+2],
                                                 self.board_position[row_id][i+3])
                sum1 = min(sum1, strip_eval[0])
                pot1 = pot1 or strip_eval[1]
                sum2 = max(sum2, strip_eval[2])
                pot2 = pot2 or strip_eval[3]
                i = i + 1

        # check diagonal top-left bottom-right
        for row in range(center_height):
            row_id = self.ROW_COUNT - 1 - row
            for d in range(max(0, row-2), min(row+1, 4), 1):
                strip_eval = self.evaluate_strip(self.board_position[row_id-3+d][d],
                                                 self.board_position[row_id-2+d][d+1],
                                                 self.board_position[row_id-1+d][d+2],
                                                 self.board_position[row_id+d][d+3])
                sum1 = min(sum1, strip_eval[0])
                pot1 = pot1 or strip_eval[1]
                sum2 = max(sum2, strip_eval[2])
                pot2 = pot2 or strip_eval[3]

        # check diagonal bottom-left top-right
        for row in range(center_height):
            row_id = self.ROW_COUNT - 1 - row
            for d in range(max(0, 3-row), min(6-row, 4), 1):
                strip_eval = self.evaluate_strip(self.board_position[row_id-d+3][d],
                                                 self.board_position[row_id-d+2][d+1],
                                                 self.board_position[row_id-d+1][d+2],
                                                 self.board_position[row_id-d][d+3])
                sum1 = min(sum1, strip_eval[0])
                pot1 = pot1 or strip_eval[1]
                sum2 = max(sum2, strip_eval[2])
                pot2 = pot2 or strip_eval[3]
        return [sum1, pot1, sum2, pot2]

    def first_empty_row(self, board_column_id):
        for row in range(self.ROW_COUNT-1, -1, -1):
            if self.board_position[row][board_column_id] == 0:
                return row
        return -1

    def play_piece(self, row, column):
        # assign -1 for player 1 and +1 for player 2
        self.board_position[row][column] = (self.current_player - 1.5) * 2
        self.pieces_played = self.pieces_played + 1

        if self.announce_winner:
            evaluation = self.evaluate_board()
            if abs(evaluation[0]) == 4 or abs(evaluation[2]) == 4:
                self.winner = self.current_player
                self.current_state = self.GAME_OVER
            elif self.pieces_played == self.ROW_COUNT * self.COLUMN_COUNT:
                self.current_state = self.GAME_OVER

        # switch player
        self.previous_player = self.current_player
        self.current_player = 3 - self.current_player
