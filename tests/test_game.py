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



class TestUncoverTrapEdgeCases(unittest.TestCase):
    def test_uncover_opponent_win_detection(self):
        state = GameState(win_streak=4)
        # Setup row 0 with 3 Orange pieces and 1 Blue piece covering an Orange piece
        state.board[0][0] = [Piece(PLAYER_ORANGE, 2)]
        state.board[0][1] = [Piece(PLAYER_ORANGE, 2)]
        state.board[0][2] = [Piece(PLAYER_ORANGE, 2)]
        # Cell (0, 3) has Orange buried under Blue Large
        state.board[0][3] = [Piece(PLAYER_ORANGE, 1), Piece(PLAYER_BLUE, 3)]

        # If Blue lifts piece at (0, 3), Orange instantly has 4 in a row!
        is_trap, lines = state.would_lift_uncover_opponent_win((0, 3))
        self.assertTrue(is_trap)
        self.assertEqual(len(lines), 1)

if __name__ == '__main__':
    unittest.main()
