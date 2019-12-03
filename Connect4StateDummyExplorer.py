import ast
import numpy as np

class GamePrinter():
    def __init__(self, state_list_file):
        self.board_sequence = []
        f = open(state_list_file, "r")
        if f.mode == "r":
            lines = f.readlines()
            for l in lines:
                self.board_sequence.append(ast.literal_eval(l[:-1]))
        f.close()
        self.current_state_id = 0

    def print_game(self):
        for i in range(len(self.board_sequence)):
            print(i)
            print(np.array(self.board_sequence[i]))

# main method
def main():
    gp = GamePrinter(state_list_file="c4games/duel_15x1002_game1.txt")
    gp.print_game()
    gp = GamePrinter(state_list_file="c4games/duel_15x1002_game2.txt")
    gp.print_game()

if __name__ == "__main__":
    main()
