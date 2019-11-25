class Connect4Game():
    ROW_COUNT = 6
    COLUMN_COUNT = 7

    # game "states"
    GAME_RUNNING = 1
    GAME_OVER = 2

    def __init__(self, announce_winner=False):
        self.current_player = 1
        self.board_position = self.new_board()

        self.announce_winner = announce_winner
        self.current_state = self.GAME_RUNNING
        self.winner = None

    # @property
    # def current_player(self):
    #     return self.__current_player
    #
    # @property
    # def current_state(self):
    #     return self.__current_state
    #
    # @property
    # def winner(self):
    #     return self.__winner
    #
    # @property
    # def board_position(self):
    #     return self.__board_position

    def new_board(self):
        # Create the 'empty' board of 0's
        board = [[0 for _x in range(self.COLUMN_COUNT)] for _y in range(self.ROW_COUNT)]
        return board

    def get_winner(self):
        center_height = self.ROW_COUNT - 1 - self.first_empty_row(3)
        w = None
        # check vertical
        for column in range(self.COLUMN_COUNT):
            height = self.ROW_COUNT - 1 - self.first_empty_row(column)
            i = 0
            while i <= height - 4:
                row_id = self.ROW_COUNT - 1 - i
                if self.board_position[row_id][column] == \
                        self.board_position[row_id-1][column] and self.board_position[row_id][column] == \
                        self.board_position[row_id-2][column] and self.board_position[row_id][column] == \
                        self.board_position[row_id-3][column]:
                    w = self.board_position[row_id][column]
                i = i + 1

        # check horizontal
        for row in range(center_height):
            row_id = self.ROW_COUNT - 1 - row
            i = 0
            while i < 4:
                if self.board_position[row_id][i] == \
                        self.board_position[row_id][i+1] and self.board_position[row_id][i] == \
                        self.board_position[row_id][i+2] and self.board_position[row_id][i] == \
                        self.board_position[row_id][i+3]:
                    w = self.board_position[row_id][i]
                i = i + 1

        # check diagonal top-left bottom-right
        for row in range(center_height):
            row_id = self.ROW_COUNT - 1 - row
            for d in range(max(0, row-2), min(row+1, 4), 1):
                if self.board_position[row_id-3+d][d] == \
                        self.board_position[row_id-2+d][d+1] and self.board_position[row_id-3+d][d] == \
                        self.board_position[row_id-1+d][d+2] and self.board_position[row_id-3+d][d] == \
                        self.board_position[row_id+d][d+3]:
                    w = self.board_position[row_id-3+d][d]

        # check diagonal bottom-left top-right
        for row in range(center_height):
            row_id = self.ROW_COUNT - 1 - row
            for d in range(max(0, 3-row), min(6-row, 4), 1):
                if self.board_position[row_id-d+3][d] == \
                        self.board_position[row_id-d+2][d+1] and self.board_position[row_id-d+3][d] == \
                        self.board_position[row_id-d+1][d+2] and self.board_position[row_id-d+3][d] == \
                        self.board_position[row_id-d][d+3]:
                    w = self.board_position[row_id-d+3][d]
        return w

    def first_empty_row(self, board_column_id):
        for row in range(self.ROW_COUNT-1, -1, -1):
            if self.board_position[row][board_column_id] == 0:
                return row
        return -1

    def play_piece(self, row, column):
        # assign -1 for player 1 and +1 for player 2
        self.board_position[row][column] = (self.current_player - 1.5) * 2

        # switch player
        self.current_player = 3 - self.current_player

        if self.announce_winner:
            self.winner = self.get_winner()
            if self.winner is not None:
                self.current_state = self.GAME_OVER
