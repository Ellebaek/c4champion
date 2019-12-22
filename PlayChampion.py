import tensorflow as tf
from Connect4GameWindow import Connect4GameWindow
from DeepC4AgentTF import DeepC4AgentTF
from qlearning_helper import open_actions, get_state, best_allowed_action
from Connect4Game import Connect4Game
import arcade
#import ast

WINDOW_WIDTH = 700

ckpt_dir = 'checkpoints/'


class PlayChampion(Connect4GameWindow):
    def __init__(self, tf_session, champion_agent, champion_player_id, window_title):
        super().__init__(WINDOW_WIDTH, announce_winner=True, title=window_title)
        self.board_sequence = []
        self.champion_player_id = champion_player_id
        self.champion_agent = champion_agent
        self.sess = tf_session

    def on_update(self, delta_time: float):
        if self.current_game.current_player == self.champion_player_id and self.current_game.current_state == Connect4Game.GAME_RUNNING:
            # CHAMPION TO PLAY
            s = get_state(self.current_game)
            filter = open_actions(self.current_game)
            allQ = self.sess.run(self.champion_agent.Qout, feed_dict={self.champion_agent.inputs: s, self.champion_agent.keep_pct: 1})
            a = best_allowed_action(allQ, filter, 1)
            self.play_piece(self.current_game.first_empty_row(a), a)


# main method
def main():
    CHAMPION = DeepC4AgentTF("Agent1")
    saver = tf.train.Saver()
    # vars_global = tf.global_variables()

    with tf.compat.v1.Session() as sess:
        # sess.run(init)
        saver.restore(sess, "{0}dr_test1_0.ckpt".format(ckpt_dir))
        ep = sess.run(CHAMPION.training_episodes, feed_dict={})
#        print("Playing CHAMPION: {0} trained for {1} episodes".format(CHAMPION.name, ep))
        PlayChampion(tf_session=sess,
                     champion_agent=CHAMPION,
                     champion_player_id=2,
                     window_title="Playing against champion agent trained for {0:.0f} episodes".format(ep))
        arcade.run()


if __name__ == "__main__":
    main()
