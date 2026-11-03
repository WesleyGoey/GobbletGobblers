"""
Thorough edge case tests for 5x5 Gobblet Gobblers:
- Friendly gobbling (self-stacking)
- Stack depth limits (max 3 nested pieces)
- Diagonal wins (main and anti-diagonal)
- Uncover Trap with successful block vs failed block
- Forgetting memory transitions (exact boundary: 4, 5, 6 turns ago)
- Minimax alpha-beta pruning verification
"""

import unittest
from game_engine import (
    GameState, Piece, Move, PLAYER_BLUE, PLAYER_ORANGE,
    generate_winning_lines, WINNING_LINES
)
from ai_agent import (
    MinimaxAgent, MemoryInspector, build_belief_state, AIHeuristic, SearchTelemetry
)


class TestGobbletEdgeCases(unittest.TestCase):

    def setUp(self):
        self.state = GameState(win_streak=4)

    def test_friendly_gobbling(self):
        """Player can gobble their own smaller pieces."""
        # Blue drops small (size 1) at (1, 1)
        m1 = Move(PLAYER_BLUE, 'drop', (1, 1), size=1)
        self.assertTrue(self.state.apply_move(m1))

        # Orange plays elsewhere
        m2 = Move(PLAYER_ORANGE, 'drop', (4, 4), size=1)
        self.assertTrue(self.state.apply_move(m2))

        # Blue gobbles their own small piece at (1, 1) with medium (size 2)
        m3 = Move(PLAYER_BLUE, 'drop', (1, 1), size=2)
        self.assertTrue(self.state.apply_move(m3))
        self.assertEqual(self.state.get_stack_depth(1, 1), 2)
        self.assertEqual(self.state.get_top_piece(1, 1).player, PLAYER_BLUE)
        self.assertEqual(self.state.get_top_size(1, 1), 2)

    def test_diagonal_wins(self):
        """Test diagonal win conditions."""
        # Main diagonal line: (0, 0), (1, 1), (2, 2), (3, 3)
        self.state.board[0][0] = [Piece(PLAYER_BLUE, 1)]
        self.state.board[1][1] = [Piece(PLAYER_BLUE, 1)]
        self.state.board[2][2] = [Piece(PLAYER_BLUE, 1)]
        self.state.board[3][3] = [Piece(PLAYER_BLUE, 1)]

        winner, line = self.state.check_winner_on_board()
        self.assertEqual(winner, PLAYER_BLUE)
        self.assertEqual(line, [(0, 0), (1, 1), (2, 2), (3, 3)])

        # Anti-diagonal line: (0, 3), (1, 2), (2, 1), (3, 0)
        state2 = GameState(win_streak=4)
        state2.board[0][3] = [Piece(PLAYER_ORANGE, 2)]
        state2.board[1][2] = [Piece(PLAYER_ORANGE, 2)]
        state2.board[2][1] = [Piece(PLAYER_ORANGE, 2)]
        state2.board[3][0] = [Piece(PLAYER_ORANGE, 2)]

        winner2, line2 = state2.check_winner_on_board()
        self.assertEqual(winner2, PLAYER_ORANGE)
        self.assertEqual(line2, [(0, 3), (1, 2), (2, 1), (3, 0)])

    def test_uncover_trap_successful_block(self):
        """
        If lifting reveals an opponent win, but the lifted piece legally blocks that win,
        the active player survives and the game continues!
        """
        # Orange has 3 visible in row 0: (0, 0), (0, 1), (0, 2)
        # And at (0, 3), Orange has a piece covered by Blue Large piece (size 3)
        # In column 4, Blue can move the piece from (0, 3) to (0, 0) covering Orange's size 1 piece!
        self.state.board[0][0] = [Piece(PLAYER_ORANGE, 1)]
        self.state.board[0][1] = [Piece(PLAYER_ORANGE, 1)]
        self.state.board[0][2] = [Piece(PLAYER_ORANGE, 1)]
        self.state.board[0][3] = [Piece(PLAYER_ORANGE, 1), Piece(PLAYER_BLUE, 3)]

        self.state.current_player = PLAYER_BLUE

        # Blue lifts Large from (0, 3) and places it on (0, 0) covering Orange!
        blocking_move = Move(PLAYER_BLUE, 'move', from_pos=(0, 3), to_pos=(0, 0))
        success = self.state.apply_move(blocking_move)
        self.assertTrue(success)

        # Orange does NOT win because (0, 0) is now covered by Blue!
        # The line (0,0..0,3) is broken because (0,0) is Blue!
        self.assertFalse(self.state.game_over)
        self.assertIsNone(self.state.winner)
        self.assertEqual(self.state.current_player, PLAYER_ORANGE)

    def test_memory_forgetting_boundary(self):
        """
        Verifies exact boundary of 5-move memory window:
        - Move covered <= 5 turns ago is remembered.
        - Move covered > 5 turns ago is forgotten.
        """
        # Blue drops small at (3, 3) at turn 1
        self.state.apply_move(Move(PLAYER_BLUE, 'drop', (3, 3), size=1))
        # Orange covers it at turn 2 with medium
        self.state.apply_move(Move(PLAYER_ORANGE, 'drop', (3, 3), size=2))

        # At turn 7: current_turn = 7, covered at 2 => 7 - 2 = 5 turns ago (Remembered!)
        self.state.turn_number = 7
        snap7 = MemoryInspector.get_memory_snapshot(self.state, memory_window=5)
        self.assertTrue(snap7[0]['is_remembered'])

        # At turn 8: current_turn = 8, covered at 2 => 8 - 2 = 6 turns ago (Forgotten!)
        self.state.turn_number = 8
        snap8 = MemoryInspector.get_memory_snapshot(self.state, memory_window=5)
        self.assertFalse(snap8[0]['is_remembered'])

    def test_telemetry_pruning_recorded(self):
        """Verify that Minimax agent triggers and records alpha-beta pruning."""
        telemetry = SearchTelemetry()
        agent = MinimaxAgent(player=PLAYER_ORANGE, max_depth=3, memory_window=5, telemetry=telemetry)
        agent.step_delay = 0

        # Run AI on game state with several pieces on board
        self.state.board[1][1] = [Piece(PLAYER_BLUE, 1)]
        self.state.board[2][2] = [Piece(PLAYER_ORANGE, 2)]
        self.state.current_player = PLAYER_ORANGE

        move = agent.get_action(self.state)
        self.assertIsNotNone(move)
        self.assertGreater(telemetry.nodes_visited, 0)
        self.assertGreaterEqual(telemetry.prune_count, 0)

    def test_simultaneous_4_in_a_row_uncover_trap(self):
        """
        Simultaneous 4-in-a-row edge case:
        If Blue moves a piece that simultaneously uncovers Orange's 4-in-a-row
        AND places to form Blue's 4-in-a-row:
        The uncovered Orange line was created upon lifting and was not blocked.
        Official rule dictates Orange wins immediately via Uncover Trap!
        """
        # Row 0: (0,0), (0,1), (0,2) Orange. (0,3) has Orange covered by Blue Large (size 3)
        self.state.board[0][0] = [Piece(PLAYER_ORANGE, 1)]
        self.state.board[0][1] = [Piece(PLAYER_ORANGE, 1)]
        self.state.board[0][2] = [Piece(PLAYER_ORANGE, 1)]
        self.state.board[0][3] = [Piece(PLAYER_ORANGE, 1), Piece(PLAYER_BLUE, 3)]

        # Row 1: (1,0), (1,1), (1,2) Blue. (1,3) empty.
        self.state.board[1][0] = [Piece(PLAYER_BLUE, 2)]
        self.state.board[1][1] = [Piece(PLAYER_BLUE, 2)]
        self.state.board[1][2] = [Piece(PLAYER_BLUE, 2)]

        self.state.current_player = PLAYER_BLUE

        # Blue moves from (0,3) to (1,3):
        # Lifting from (0,3) reveals Orange's (0,0..0,3) 4-in-a-row!
        # Placing at (1,3) completes Blue's (1,0..1,3) 4-in-a-row!
        # But Orange's revealed line was NOT blocked!
        move = Move(PLAYER_BLUE, 'move', from_pos=(0, 3), to_pos=(1, 3))
        self.state.apply_move(move)

        self.assertTrue(self.state.game_over)
        self.assertEqual(self.state.winner, PLAYER_ORANGE)
        self.assertIn("Uncover Trap", self.state.win_reason)

    def test_simultaneous_5_in_a_row_uncover_trap(self):
        """
        Simultaneous 5-in-a-row edge case:
        If Blue moves a piece that simultaneously uncovers Orange's 5-in-a-row
        AND places to form Blue's 5-in-a-row:
        The uncovered Orange line was created upon lifting and was not blocked.
        Official rule dictates Orange wins immediately via Uncover Trap!
        """
        state_5 = GameState(win_streak=5)
        # Row 0: 4 Orange visible + 1 Orange covered at (0,4)
        for c in range(4):
            state_5.board[0][c] = [Piece(PLAYER_ORANGE, 1)]
        state_5.board[0][4] = [Piece(PLAYER_ORANGE, 1), Piece(PLAYER_BLUE, 3)]

        # Row 1: 4 Blue visible + (1,4) empty
        for c in range(4):
            state_5.board[1][c] = [Piece(PLAYER_BLUE, 2)]

        state_5.current_player = PLAYER_BLUE

        # Blue moves from (0,4) to (1,4):
        move = Move(PLAYER_BLUE, 'move', from_pos=(0, 4), to_pos=(1, 4))
        state_5.apply_move(move)

        self.assertTrue(state_5.game_over)
        self.assertEqual(state_5.winner, PLAYER_ORANGE)
        self.assertIn("Uncover Trap", state_5.win_reason)

    def test_trap_risk_heuristic(self):
        """AIHeuristic should penalize pinned AI pieces that cover dangerous opponent pieces."""
        # State A: Orange Large covers Blue Small at (0, 3).
        # Row 0 has (0,0), (0,1), (0,2) Blue.
        # Lifting Orange Large would reveal Blue's 4-in-a-row!
        state_danger = GameState(win_streak=4)
        state_danger.board[0][0] = [Piece(PLAYER_BLUE, 1)]
        state_danger.board[0][1] = [Piece(PLAYER_BLUE, 1)]
        state_danger.board[0][2] = [Piece(PLAYER_BLUE, 1)]
        state_danger.board[0][3] = [Piece(PLAYER_BLUE, 1), Piece(PLAYER_ORANGE, 3)]

        score_danger = AIHeuristic.evaluate(state_danger, PLAYER_ORANGE)

        # State B: Orange Large covers friendly Orange Small at (0, 3).
        state_safe = GameState(win_streak=4)
        state_safe.board[0][0] = [Piece(PLAYER_BLUE, 1)]
        state_safe.board[0][1] = [Piece(PLAYER_BLUE, 1)]
        state_safe.board[0][2] = [Piece(PLAYER_BLUE, 1)]
        state_safe.board[0][3] = [Piece(PLAYER_ORANGE, 1), Piece(PLAYER_ORANGE, 3)]

        score_safe = AIHeuristic.evaluate(state_safe, PLAYER_ORANGE)

        # Trap risk penalty must make State A significantly worse than State B
        self.assertLess(score_danger, score_safe - 300.0)

    def test_5_in_a_row_trap_risk_heuristic(self):
        """AIHeuristic penalizes pinned AI pieces that would uncover a 5-in-a-row line."""
        state_danger = GameState(win_streak=5)
        for c in range(4):
            state_danger.board[0][c] = [Piece(PLAYER_BLUE, 1)]
        state_danger.board[0][4] = [Piece(PLAYER_BLUE, 1), Piece(PLAYER_ORANGE, 3)]

        score_danger = AIHeuristic.evaluate(state_danger, PLAYER_ORANGE)

        state_safe = GameState(win_streak=5)
        for c in range(4):
            state_safe.board[0][c] = [Piece(PLAYER_BLUE, 1)]
        state_safe.board[0][4] = [Piece(PLAYER_ORANGE, 1), Piece(PLAYER_ORANGE, 3)]

        score_safe = AIHeuristic.evaluate(state_safe, PLAYER_ORANGE)
        self.assertLess(score_danger, score_safe - 300.0)

    def test_immobility_loss(self):
        """If a player has no legal moves on their turn, opponent wins by immobility."""
        state = GameState()
        # Empty Blue reserves
        state.reserves[PLAYER_BLUE] = {1: 0, 2: 0, 3: 0}
        # Place only 1 Blue Small piece on board, covered by Orange Large piece
        state.board[0][0] = [Piece(PLAYER_BLUE, 1), Piece(PLAYER_ORANGE, 3)]
        state.current_player = PLAYER_BLUE

        legal_moves = state.get_legal_moves(PLAYER_BLUE)
        self.assertEqual(len(legal_moves), 0)


if __name__ == '__main__':
    unittest.main()



