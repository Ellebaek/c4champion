import arcade
import PIL
from PIL import ImageDraw
from Connect4Game import Connect4Game

WINDOW_WIDTH = 700
MARGIN_PERCENTAGE = 0.1
# ROW_COUNT = 6
# COLUMN_COUNT = 7

# game "states"
# GAME_RUNNING = 1
# GAME_OVER = 2

# first color is no piece played (0)
# second color is player one (-1)
# third color is player two (1)
colors = [
    (255, 255, 255),
    (255, 0, 0),
    (255, 255, 0)
]


class Connect4GameWindow(arcade.Window):
    def __init__(self, width, announce_winner=False):
        height = int(width / Connect4Game.COLUMN_COUNT * Connect4Game.ROW_COUNT)
        super().__init__(width, height, 'Connect4 Game Window')
        self.set_location(200, 100)
        self.left_margin = self.width * MARGIN_PERCENTAGE / 2
        self.top_margin = self.height * MARGIN_PERCENTAGE / 2
        self.bottom_margin = self.height * MARGIN_PERCENTAGE / 2
        self.circle_max_radius = self.width * (1 - MARGIN_PERCENTAGE) / (Connect4Game.COLUMN_COUNT * 2)

        self.texture_list = self.create_textures()
        self.player_position = 4
        self.current_game = None
        #self.current_player = 1
        # self.board_position = None
        self.shape_list = None
        self.board_sprite_list = None

        self.announce_winner = announce_winner
        # self.winner = None
        self.setup()

    def create_textures(self):
        """ Create a list of images for sprites based on the global colors. """
        new_textures = []
        for color in colors:
            image = PIL.Image.new('RGBA', (int(self.circle_max_radius * 2), int(self.circle_max_radius * 2)),
                                  (255, 255, 255, 0))
            draw = ImageDraw.Draw(image)
            draw.ellipse((self.circle_max_radius * 2 * 0.1, self.circle_max_radius * 2 * 0.1,
                          self.circle_max_radius * 2 * 0.9, self.circle_max_radius * 2 * 0.9), fill=color)
            new_textures.append(arcade.Texture(str(color), image=image))
        return new_textures

    def setup(self):
        # create new game
        self.current_game = Connect4Game(announce_winner=True)

        # reset player position, current player and board
        self.player_position = 4

        # create shapes for drawing background board
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

        # create sprites (all potential circles and colors)
        self.board_sprite_list = arcade.SpriteList()
        for row in range(self.current_game.ROW_COUNT):
            for column in range(self.current_game.COLUMN_COUNT):
                sprite = arcade.Sprite()
                # assign all three color options to each field
                for texture in self.texture_list:
                    sprite.append_texture(texture)
                # set default / start color
                sprite.set_texture(0)
                sprite.center_x = self.left_margin + (self.circle_max_radius * (2 * column + 1))
                sprite.center_y = self.height - self.bottom_margin - (self.circle_max_radius * (2 * row + 1))

                self.board_sprite_list.append(sprite)

    def on_draw(self):
        # This command has to happen before we start drawing
        arcade.start_render()

        if self.current_game.current_state == Connect4Game.GAME_RUNNING:
            self.draw_running_game()
        else:
            self.draw_running_game()
            self.draw_game_over()

    def draw_running_game(self):
        self.shape_list.draw()
        self.board_sprite_list.draw()

        # draw cursor
        cursor_position = self.width / 2 + (self.player_position - 4) * self.circle_max_radius * 2
        arcade.draw_triangle_filled(cursor_position - self.circle_max_radius, self.height,
                                    cursor_position + self.circle_max_radius, self.height,
                                    cursor_position, self.height - self.top_margin,
                                    colors[self.current_game.current_player])

    def draw_game_over(self):
        arcade.draw_text("Game Over",
                         self.width / 2 - 150,
                         self.height / 2,
                         arcade.color.BLACK_OLIVE, 54)
        arcade.draw_text("Press ENTER to restart",
                         self.width / 2 - 150,
                         self.height / 2 - 50,
                         arcade.color.BLACK_OLIVE, 28)

    def get_winner(self):
        center_height = Connect4Game.ROW_COUNT - 1 - self.current_game.first_empty_row(3)
        w = None
        # check vertical
        for column in range(Connect4Game.COLUMN_COUNT):
            height = Connect4Game.ROW_COUNT - 1 - self.current_game.first_empty_row(column)
            i = 0
            while i <= height - 4:
                row_id = Connect4Game.ROW_COUNT - 1 - i
                if self.board_position[row_id][column] == \
                        self.board_position[row_id - 1][column] and self.board_position[row_id][column] == \
                        self.board_position[row_id - 2][column] and self.board_position[row_id][column] == \
                        self.board_position[row_id - 3][column]:
                    w = self.board_position[row_id][column]
                i = i + 1

        # check horizontal
        for row in range(center_height):
            row_id = Connect4Game.ROW_COUNT - 1 - row
            i = 0
            while i < 4:
                if self.board_position[row_id][i] == \
                        self.board_position[row_id][i + 1] and self.board_position[row_id][i] == \
                        self.board_position[row_id][i + 2] and self.board_position[row_id][i] == \
                        self.board_position[row_id][i + 3]:
                    w = self.board_position[row_id][i]
                i = i + 1

        # check diagonal top-left bottom-right
        for row in range(center_height):
            row_id = Connect4Game.ROW_COUNT - 1 - row
            for d in range(max(0, row - 2), min(row + 1, 4), 1):
                if self.board_position[row_id - 3 + d][d] == \
                        self.board_position[row_id - 2 + d][d + 1] and self.board_position[row_id - 3 + d][d] == \
                        self.board_position[row_id - 1 + d][d + 2] and self.board_position[row_id - 3 + d][d] == \
                        self.board_position[row_id + d][d + 3]:
                    w = self.board_position[row_id - 3 + d][d]

        # check diagonal bottom-left top-right
        for row in range(center_height):
            row_id = Connect4Game.ROW_COUNT - 1 - row
            for d in range(max(0, 3 - row), min(6 - row, 4), 1):
                if self.board_position[row_id - d + 3][d] == \
                        self.board_position[row_id - d + 2][d + 1] and self.board_position[row_id - d + 3][d] == \
                        self.board_position[row_id - d + 1][d + 2] and self.board_position[row_id - d + 3][d] == \
                        self.board_position[row_id - d][d + 3]:
                    w = self.board_position[row_id - d + 3][d]
        return w

    # def first_empty_row(self, board_column_id):
    #     for row in range(Connect4Game.ROW_COUNT-1,-1,-1):
    #         if self.board_position[row][board_column_id] == 0:
    #             return row
    #     return -1

    def on_key_press(self, symbol, modifiers):
        if symbol == arcade.key.RIGHT and self.current_game.current_state == Connect4Game.GAME_RUNNING:
            self.player_position = min(self.player_position + 1, Connect4Game.COLUMN_COUNT)
        elif symbol == arcade.key.LEFT and self.current_game.current_state == Connect4Game.GAME_RUNNING:
            self.player_position = max(self.player_position - 1, 1)
        elif symbol == arcade.key.DOWN and self.current_game.current_state == Connect4Game.GAME_RUNNING:
            open_row = self.current_game.first_empty_row(self.player_position - 1)
            if open_row > -1:
                self.play_piece(open_row, self.player_position - 1)
        elif symbol == arcade.key.ENTER:  # and self.current_state == GAME_OVER:
            # Restart the game.
            self.setup()

    def play_piece(self, row, column):
        # update UI
        i = row * Connect4Game.COLUMN_COUNT + column
        self.board_sprite_list[i].set_texture(self.current_game.current_player)

        # update underlying game and switch player
        self.current_game.play_piece(row, column)

# main method
def main():
    Connect4GameWindow(WINDOW_WIDTH,
                       announce_winner=True)
    arcade.run()


if __name__ == "__main__":
    main()
