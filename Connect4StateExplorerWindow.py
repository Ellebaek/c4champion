from Connect4GameWindow import Connect4GameWindow
from Connect4Game import Connect4Game
import arcade
import ast

WINDOW_WIDTH = 700


class Connect4StateExplorerWindow(Connect4GameWindow):
    def __init__(self, state_list_file):
        super().__init__(WINDOW_WIDTH, announce_winner=False)
        self.board_sequence = []
        f = open(state_list_file, "r")
        if f.mode == "r":
            lines = f.readlines()
            for l in lines:
                self.board_sequence.append(ast.literal_eval(l[:-1]))
        f.close()
        self.current_state_id = 0

    def on_draw(self):
        # This command has to happen before we start drawing
        arcade.start_render()
        self.shape_list.draw()
        self.board_sprite_list.draw()
        arcade.draw_text("{0} of {1}".format(self.current_state_id, len(self.board_sequence)-1),
                         self.width / 2 - 100,
                         self.height - 25,
                         arcade.color.BLACK, 20)

    def on_key_press(self, symbol, modifiers):
        if symbol == arcade.key.RIGHT:
            self.current_state_id = min(self.current_state_id + 1, len(self.board_sequence)-1)
        elif symbol == arcade.key.LEFT:
            self.current_state_id = max(self.current_state_id - 1, 0)

    def on_update(self, delta_time: float):
        board = self.board_sequence[self.current_state_id]
        for row in range(Connect4Game.ROW_COUNT):
            for column in range(Connect4Game.COLUMN_COUNT):
                i = row * Connect4Game.COLUMN_COUNT + column
                if board[row][column] != 0:
                    self.board_sprite_list[i].set_texture(int((board[row][column] + 3) / 2))
                else:
                    self.board_sprite_list[i].set_texture(0)

    def play_piece(self, row, column):
        pass


# main method
def main():
    Connect4StateExplorerWindow(state_list_file="c4games/duel_7x500_game2.txt")
    arcade.run()


if __name__ == "__main__":
    main()
