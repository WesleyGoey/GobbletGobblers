import unittest
from game_engine import GameState, Move, Piece, PLAYER_BLUE, PLAYER_ORANGE


class TestGobbletEngine(unittest.TestCase):
    def setUp(self):
        self.state = GameState(win_streak=5)

    def test_initial_state(self):
        self.assertEqual(len(self.state.board), 5)
        self.assertEqual(self.state.current_player, PLAYER_BLUE)
        self.assertEqual(self.state.reserves[PLAYER_BLUE][3], 4)
        self.assertFalse(self.state.game_over)

    def test_valid_drop_and_gobble(self):
        # Blue drops Medium (2) at (2, 2)
        m1 = Move(PLAYER_BLUE, 'drop', to_pos=(2, 2), size=2)
        self.assertTrue(self.state.apply_move(m1))
        self.assertEqual(self.state.get_top_size(2, 2), 2)

        # Orange gobbles with Large (3) at (2, 2)
        m2 = Move(PLAYER_ORANGE, 'drop', to_pos=(2, 2), size=3)
        self.assertTrue(self.state.apply_move(m2))
        self.assertEqual(self.state.get_top_piece(2, 2).player, PLAYER_ORANGE)
        self.assertEqual(self.state.get_stack_depth(2, 2), 2)

    def test_illegal_gobble_smaller_over_larger(self):
        # Place Large first
        self.state.apply_move(Move(PLAYER_BLUE, 'drop', to_pos=(0, 0), size=3))
        # Orange attempts to drop Small over Large -> Must fail
        m_illegal = Move(PLAYER_ORANGE, 'drop', to_pos=(0, 0), size=1)
        self.assertFalse(self.state.apply_move(m_illegal))


if __name__ == '__main__':
    unittest.main()
