import arcade
import copy

WINDOW_WIDTH = 700
MARGIN_PERCENTAGE = 0.1
# game "states"
GAME_RUNNING = 1
GAME_OVER = 2

class Connect4Game(arcade.Window):
    def __init__(self, width, height, title, start_position=[[], [], [], [], [], [], []], announce_winner=False):
        super().__init__(width, height, title)
        self.set_location(200, 100)
        self.left_margin = self.width * MARGIN_PERCENTAGE / 2
        self.top_margin = self.height * MARGIN_PERCENTAGE / 2
        self.bottom_margin = self.height * MARGIN_PERCENTAGE / 2
        self.circle_max_radius = self.width * (1 - MARGIN_PERCENTAGE) / (7 * 2)
        self.player_position = 4
        self.player_color = arcade.color.RED
        self.start_position = start_position
        self.board_position = copy.deepcopy(start_position)
        self.shape_list = None
        self.announce_winner = announce_winner
        self.current_state = GAME_RUNNING
        self.winner = None
        self.setup()

    def setup(self):
        arcade.set_background_color(arcade.color.WHITE)
        self.shape_list = arcade.ShapeElementList()

        self.shape_list.append(
            arcade.create_rectangle_filled(self.width / 2, self.height / 2,
                                           self.width * (1 - MARGIN_PERCENTAGE + 0.02),
                                           self.height * (1 - MARGIN_PERCENTAGE + 0.02),
                                           arcade.color.DARK_PASTEL_BLUE)
        )

        self.shape_list.append(
            arcade.create_rectangle_filled(self.width / 2, self.height / 2,
                                           self.width * (1 - MARGIN_PERCENTAGE),
                                           self.height * (1 - MARGIN_PERCENTAGE),
                                           arcade.color.BRIGHT_NAVY_BLUE)
        )

        for l in range(7):
            for b in range(6):
                self.shape_list.append(
                    arcade.create_ellipse_filled(self.left_margin + (self.circle_max_radius * (2 * l + 1)),
                                                 self.bottom_margin + (self.circle_max_radius * (2 * b + 1)),
                                                 self.circle_max_radius * 0.85,
                                                 self.circle_max_radius * 0.85,
                                                 arcade.color.WHITE)
                )

    def on_draw(self):
        # This command has to happen before we start drawing
        arcade.start_render()

        if self.current_state == GAME_RUNNING:
            self.draw_running_game()
        else:
            self.draw_running_game()
            self.draw_game_over()

    def draw_running_game(self):
        self.shape_list.draw()

        # draw played board
        for l in range(len(self.board_position)):
            for b in range(len(self.board_position[l])):
                arcade.draw_circle_filled(self.left_margin + (self.circle_max_radius * (2 * l + 1)),
                                          self.bottom_margin + (self.circle_max_radius * (2 * b + 1)),
                                          self.circle_max_radius * 0.85,
                                          self.get_color(l + 1, b + 1))

        # draw cursor
        cursor_position = self.width / 2 + (self.player_position - 4) * self.circle_max_radius * 2
        arcade.draw_triangle_filled(cursor_position - self.circle_max_radius, self.height,
                                    cursor_position + self.circle_max_radius, self.height,
                                    cursor_position, self.height - self.top_margin,
                                    self.player_color)

    def draw_game_over(self):
        arcade.draw_text("Game Over",
                         self.width/2 - 150,
                         self.height/2,
                         arcade.color.BLACK_OLIVE, 54)
        arcade.draw_text("Press ENTER to restart",
                         self.width/2 - 150,
                         self.height/2 - 50,
                         arcade.color.BLACK_OLIVE, 28)

    def get_winner(self):
        temp_board = copy.deepcopy(self.board_position)
        w = None
        col_heights = []
        # check vertical
        for l in temp_board:
            col_heights.append(len(l))
            i = 0
            while i <= len(l) - 4:
                if l[i] == l[i + 1] and l[i] == l[i + 2] and l[i] == l[i + 3]:
                    w = l[i]
                i = i + 1
            # fill list with 'W' to easy horizontal and diagonal checks
            for j in range(len(l),6):
                l.append('W')

        # check horizontal
        for b in range(col_heights[3]):
            i = 0
            while i < 4:
                if temp_board[i][b] == \
                        temp_board[i+1][b] and temp_board[i][b] == \
                        temp_board[i+2][b] and temp_board[i][b] == \
                        temp_board[i+3][b]:
                    w = temp_board[i][b]
                i = i + 1

        # check diagonal top-left bottom-right
        for b in range(col_heights[3]):
            for d in range(max(0, b-2), min(b+1, 4), 1):
                if temp_board[d][b+3-d] == \
                        temp_board[d+1][b+2-d] and temp_board[d][b+3-d] == \
                        temp_board[d+2][b+1-d] and temp_board[d][b+3-d] == \
                        temp_board[d+3][b-d]:
                    w = temp_board[d][b+3-d]

        # check diagonal bottom-left top-right
        for b in range(col_heights[3]):
            for d in range(max(0, 3-b), min(6-b, 4), 1):
                if temp_board[d][b+d-3] == \
                        temp_board[d+1][b+d-2] and temp_board[d][b+d-3] == \
                        temp_board[d+2][b+d-1] and temp_board[d][b+d-3] == \
                        temp_board[d+3][b+d]:
                    w = temp_board[d][b+d-3]
        return w

    def on_key_press(self, symbol, modifiers):
        if symbol == arcade.key.RIGHT and self.current_state == GAME_RUNNING:
            self.player_position = min(self.player_position + 1, 7)
        elif symbol == arcade.key.LEFT and self.current_state == GAME_RUNNING:
            self.player_position = max(self.player_position - 1, 1)
        elif symbol == arcade.key.DOWN and self.current_state == GAME_RUNNING:
            if len(self.board_position[self.player_position - 1]) < 6:
                self.play_piece()
        elif symbol == arcade.key.ENTER: # and self.current_state == GAME_OVER:
            # Restart the game.
            self.current_state = GAME_RUNNING
            self.board_position = copy.deepcopy(self.start_position)
            self.winner = None
            self.setup()

    def play_piece(self):
        if self.player_color == arcade.color.RED:
            self.board_position[self.player_position - 1].append("R")
            self.player_color = arcade.color.YELLOW
        else:
            self.board_position[self.player_position - 1].append("Y")
            self.player_color = arcade.color.RED

        if self.announce_winner:
            self.winner = self.get_winner()
            if self.winner is not None:
                self.current_state = GAME_OVER

    def get_color(self, column_id, element_from_below):
        column = self.board_position[column_id - 1]
        if element_from_below > len(column):
            return arcade.color.WHITE
        else:
            return self.get_color_from_char(column[element_from_below - 1])

    def get_color_from_char(self, character_id) -> arcade.color:
        if character_id == "Y":
            return arcade.color.YELLOW
        elif character_id == "R":
            return arcade.color.RED
        else:
            return arcade.color.WHITE

def main():
    starting_position = [[], [], [], [], [], [], []]
    Connect4Game(WINDOW_WIDTH, int(WINDOW_WIDTH / 7 * 6),
                 'Connect4 Game Window',
                 start_position=starting_position,
                 announce_winner=True)
    arcade.run()


if __name__ == "__main__":
    main()
