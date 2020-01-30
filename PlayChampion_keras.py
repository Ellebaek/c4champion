import tensorflow as tf
from Connect4GameWindow import Connect4GameWindow
from DeepC4Agent_keras import DeepC4Agent
from qlearning_helper import open_actions, get_state, best_allowed_action
from Connect4Game import Connect4Game
import arcade

WINDOW_WIDTH = 700


class PlayChampion(Connect4GameWindow):
    def __init__(self, champion_agent, champion_player_id, window_title):
        super().__init__(WINDOW_WIDTH, announce_winner=True, title=window_title)
        self.board_sequence = []
        self.champion_player_id = champion_player_id
        self.champion_agent = champion_agent

    def on_update(self, delta_time: float):
        if self.current_game.current_player == self.champion_player_id and self.current_game.current_state == Connect4Game.GAME_RUNNING:
            # CHAMPION TO PLAY
            s = get_state(self.current_game)
            filter = open_actions(self.current_game)
            allQ = self.champion_agent.target_network.model.predict(s)
            a = best_allowed_action(allQ, filter, 1)
            self.play_piece(self.current_game.first_empty_row(a), a)


# main method
def main():
    CHAMPION = DeepC4Agent(name="AgentK5", load_models=True)

    PlayChampion(champion_agent=CHAMPION,
                 champion_player_id=2,
                 window_title="Playing champion")
    arcade.run()


if __name__ == "__main__":
    main()
