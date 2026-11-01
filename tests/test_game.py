import unittest
from game_engine import GameState, Move, Piece, PLAYER_BLUE, PLAYER_ORANGE


class TestGobbletEngine(unittest.TestCase):
    def setUp(self):
        self.state = GameState(win_streak=5)

    def test_initial_state(self):
        """Verify initial board state, reserves, and player."""
        self.assertEqual(len(self.state.board), 5)
        self.assertEqual(self.state.current_player, PLAYER_BLUE)
        self.assertEqual(self.state.reserves[PLAYER_BLUE][1], 4)
        self.assertEqual(self.state.reserves[PLAYER_BLUE][2], 4)
        self.assertEqual(self.state.reserves[PLAYER_BLUE][3], 4)
        self.assertFalse(self.state.game_over)

    def test_action_a_drop_empty_cell(self):
        """Action A: Introduce a piece from personal reserve onto an empty cell."""
        m = Move(PLAYER_BLUE, 'drop', to_pos=(2, 2), size=1)
        self.assertTrue(self.state.apply_move(m))
        self.assertEqual(self.state.get_top_size(2, 2), 1)
        self.assertEqual(self.state.reserves[PLAYER_BLUE][1], 3)
        self.assertEqual(self.state.current_player, PLAYER_ORANGE)

    def test_action_a_drop_over_smaller(self):
        """Action A: Introduce a piece over any smaller Gobbler (opponent or self)."""
        # Blue drops Size 1
        self.state.apply_move(Move(PLAYER_BLUE, 'drop', to_pos=(0, 0), size=1))
        # Orange gobbles opponent's Size 1 with Size 2
        m_orange = Move(PLAYER_ORANGE, 'drop', to_pos=(0, 0), size=2)
        self.assertTrue(self.state.apply_move(m_orange))
        self.assertEqual(self.state.get_top_piece(0, 0).player, PLAYER_ORANGE)

        # Orange turn is done, now Blue turn: Blue gobbles with Size 3
        m_blue = Move(PLAYER_BLUE, 'drop', to_pos=(0, 0), size=3)
        self.assertTrue(self.state.apply_move(m_blue))
        self.assertEqual(self.state.get_top_piece(0, 0).player, PLAYER_BLUE)
        self.assertEqual(self.state.get_stack_depth(0, 0), 3)

    def test_action_b_move_to_empty_cell(self):
        """Action B: Relocate an exposed Gobbler on the board to an empty cell."""
        # Blue drops Size 2 at (1, 1)
        self.state.apply_move(Move(PLAYER_BLUE, 'drop', to_pos=(1, 1), size=2))
        # Orange drops Size 1 at (4, 4)
        self.state.apply_move(Move(PLAYER_ORANGE, 'drop', to_pos=(4, 4), size=1))
        # Blue moves exposed Size 2 from (1, 1) to empty cell (1, 2)
        m_move = Move(PLAYER_BLUE, 'move', from_pos=(1, 1), to_pos=(1, 2), size=2)
        self.assertTrue(self.state.apply_move(m_move))
        self.assertIsNone(self.state.get_top_piece(1, 1))
        self.assertEqual(self.state.get_top_size(1, 2), 2)

    def test_action_b_move_over_smaller(self):
        """Action B: Relocate an exposed Gobbler over a smaller Gobbler on the board."""
        # Blue drops Size 1 at (2, 2)
        self.state.apply_move(Move(PLAYER_BLUE, 'drop', to_pos=(2, 2), size=1))
        # Orange drops Size 3 at (0, 0)
        self.state.apply_move(Move(PLAYER_ORANGE, 'drop', to_pos=(0, 0), size=3))
        # Blue drops Size 2 at (3, 3)
        self.state.apply_move(Move(PLAYER_BLUE, 'drop', to_pos=(3, 3), size=2))
        # Orange relocates Size 3 from (0, 0) over Blue's Size 1 at (2, 2)
        m_relocate_gobble = Move(PLAYER_ORANGE, 'move', from_pos=(0, 0), to_pos=(2, 2), size=3)
        self.assertTrue(self.state.apply_move(m_relocate_gobble))
        self.assertEqual(self.state.get_top_piece(2, 2).player, PLAYER_ORANGE)
        self.assertEqual(self.state.get_stack_depth(2, 2), 2)

    def test_action_b_restriction_cannot_move_to_same_cell(self):
        """Action B Restriction: You cannot move a piece to the exact cell it already occupies."""
        self.state.apply_move(Move(PLAYER_BLUE, 'drop', to_pos=(1, 1), size=2))
        m_same_cell = Move(PLAYER_BLUE, 'move', from_pos=(1, 1), to_pos=(1, 1), size=2)
        self.assertFalse(self.state.apply_move(m_same_cell))

    def test_russian_doll_gobble_hierarchy(self):
        """
        Russian Doll Gobble Rule:
        - Size 3 can gobble Size 2 or Size 1.
        - Size 2 can gobble Size 1.
        - Size 1 cannot gobble anything.
        """
        # Blue drops Size 1 at (0, 0)
        self.state.apply_move(Move(PLAYER_BLUE, 'drop', to_pos=(0, 0), size=1))

        # Size 1 cannot gobble Size 1 (equal size)
        m_s1_over_s1 = Move(PLAYER_ORANGE, 'drop', to_pos=(0, 0), size=1)
        self.assertFalse(self.state.apply_move(m_s1_over_s1))

        # Size 2 can gobble Size 1
        m_s2_over_s1 = Move(PLAYER_ORANGE, 'drop', to_pos=(0, 0), size=2)
        self.assertTrue(self.state.apply_move(m_s2_over_s1))

        # Blue drops Size 3 over Size 2 (Size 3 can gobble Size 2)
        m_s3_over_s2 = Move(PLAYER_BLUE, 'drop', to_pos=(0, 0), size=3)
        self.assertTrue(self.state.apply_move(m_s3_over_s2))

        # Cannot place Size 2 or Size 1 over Size 3
        self.assertFalse(self.state.apply_move(Move(PLAYER_ORANGE, 'drop', to_pos=(0, 0), size=2)))
        self.assertFalse(self.state.apply_move(Move(PLAYER_ORANGE, 'drop', to_pos=(0, 0), size=1)))


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


class TestAlphaBetaPruning(unittest.TestCase):
    def test_alpha_beta_pruning_efficiency(self):
        from ai_agent import MinimaxAgent
        state = GameState(win_streak=5)
        # Setup initial board pieces
        state.apply_move(Move(PLAYER_BLUE, 'drop', to_pos=(2, 2), size=3))

        agent = MinimaxAgent(player=PLAYER_ORANGE, max_depth=2)
        best_move = agent.get_best_move(state)
        self.assertIsNotNone(best_move)
        # Verify pruned branches were triggered and search visited fewer nodes
        self.assertGreater(agent.pruned_branches, 0)
        self.assertLess(agent.nodes_evaluated, 75 * 75)


class TestBoundedMemory(unittest.TestCase):
    def test_bounded_memory_forgetting(self):
        """Test that pieces buried longer than memory_window turns are forgotten in belief state."""
        from ai_agent import build_belief_state, MemoryInspector
        state = GameState(win_streak=5)
        # Drop Blue Small at (0, 0)
        state.apply_move(Move(PLAYER_BLUE, 'drop', to_pos=(0, 0), size=1))
        # Orange gobbles with Medium at (0, 0) on turn 2
        state.apply_move(Move(PLAYER_ORANGE, 'drop', to_pos=(0, 0), size=2))

        # Advance turn number beyond memory_window=5
        state.turn_number = 10
        belief = build_belief_state(state, memory_window=5)

        # In real state, depth is 2 (Blue Small buried)
        self.assertEqual(state.get_stack_depth(0, 0), 2)
        # In AI belief state, buried piece is forgotten (depth is 1, only top piece known)
        self.assertEqual(len(belief.board[0][0]), 1)
        self.assertEqual(belief.board[0][0][0].size, 2)

        # Memory inspector should flag piece as forgotten
        records = MemoryInspector.get_memory_snapshot(state, memory_window=5)
        self.assertTrue(any(not r['is_remembered'] for r in records))


if __name__ == '__main__':
    unittest.main()
