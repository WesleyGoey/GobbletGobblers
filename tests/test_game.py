"""
Unit tests for 5x5 Gobblet Gobblers game engine and AI agent.
"""

import unittest
from game_engine import (
    GameState, Piece, Move, PLAYER_BLUE, PLAYER_ORANGE,
    generate_winning_lines, WINNING_LINES
)
from ai_agent import (
    MinimaxAgent, MemoryInspector, build_belief_state, AIHeuristic
)


class TestGobbletGameEngine(unittest.TestCase):

    def setUp(self):
        self.state = GameState(win_streak=4)

    def test_initial_state(self):
        """Check initial board, reserves, and player."""
        self.assertEqual(self.state.current_player, PLAYER_BLUE)
        self.assertEqual(self.state.turn_number, 1)
        self.assertFalse(self.state.game_over)
        self.assertIsNone(self.state.winner)

        # Reserves: 4 of each size per player
        for p in (PLAYER_BLUE, PLAYER_ORANGE):
            self.assertEqual(self.state.reserves[p][1], 4)
            self.assertEqual(self.state.reserves[p][2], 4)
            self.assertEqual(self.state.reserves[p][3], 4)

        # Board should be completely empty
        for r in range(5):
            for c in range(5):
                self.assertEqual(self.state.get_stack_depth(r, c), 0)
                self.assertIsNone(self.state.get_top_piece(r, c))

    def test_winning_lines_count(self):
        """There should be exactly 28 winning 4-in-a-row lines on a 5x5 board."""
        self.assertEqual(len(WINNING_LINES), 28)

    def test_legal_drop_and_turn_advance(self):
        """Dropping a piece from reserve consumes from reserve and advances turn."""
        move = Move(player=PLAYER_BLUE, move_type='drop', size=3, to_pos=(2, 2))
        success = self.state.apply_move(move)
        self.assertTrue(success)
        self.assertEqual(self.state.reserves[PLAYER_BLUE][3], 3)
        self.assertEqual(self.state.current_player, PLAYER_ORANGE)
        self.assertEqual(self.state.turn_number, 2)
        self.assertEqual(self.state.get_top_size(2, 2), 3)
        self.assertEqual(self.state.get_top_piece(2, 2).player, PLAYER_BLUE)

    def test_gobble_rules(self):
        """Larger piece can gobble smaller piece; equal or smaller cannot."""
        # Blue drops small (size 1) at (0, 0)
        m1 = Move(player=PLAYER_BLUE, move_type='drop', size=1, to_pos=(0, 0))
        self.assertTrue(self.state.apply_move(m1))

        # Orange tries to drop small (size 1) at (0, 0) -> should fail (equal size)
        m2_bad = Move(player=PLAYER_ORANGE, move_type='drop', size=1, to_pos=(0, 0))
        self.assertFalse(self.state.apply_move(m2_bad))

        # Orange drops medium (size 2) at (0, 0) -> should succeed (gobble!)
        m2_good = Move(player=PLAYER_ORANGE, move_type='drop', size=2, to_pos=(0, 0))
        self.assertTrue(self.state.apply_move(m2_good))
        self.assertEqual(self.state.get_top_size(0, 0), 2)
        self.assertEqual(self.state.get_top_piece(0, 0).player, PLAYER_ORANGE)
        self.assertEqual(self.state.get_stack_depth(0, 0), 2)

        # Blue drops large (size 3) at (0, 0) -> should succeed (gobbles medium)
        m3 = Move(player=PLAYER_BLUE, move_type='drop', size=3, to_pos=(0, 0))
        self.assertTrue(self.state.apply_move(m3))
        self.assertEqual(self.state.get_stack_depth(0, 0), 3)
        self.assertEqual(self.state.get_top_size(0, 0), 3)

        # Cannot place anything more on top of size 3
        m4_bad = Move(player=PLAYER_ORANGE, move_type='drop', size=3, to_pos=(0, 0))
        self.assertFalse(self.state.apply_move(m4_bad))

    def test_win_detection_horizontal(self):
        """Player forms 4 in a row horizontally and wins."""
        # Blue drops across row 1: (1, 0), (1, 1), (1, 2), (1, 3)
        moves = [
            Move(PLAYER_BLUE, 'drop', (1, 0), size=1),
            Move(PLAYER_ORANGE, 'drop', (0, 0), size=1),
            Move(PLAYER_BLUE, 'drop', (1, 1), size=1),
            Move(PLAYER_ORANGE, 'drop', (0, 1), size=1),
            Move(PLAYER_BLUE, 'drop', (1, 2), size=1),
            Move(PLAYER_ORANGE, 'drop', (0, 2), size=1),
            Move(PLAYER_BLUE, 'drop', (1, 3), size=1),  # 4th Blue in row 1!
        ]
        for m in moves:
            self.state.apply_move(m)

        self.assertTrue(self.state.game_over)
        self.assertEqual(self.state.winner, PLAYER_BLUE)
        self.assertIn("Blue", self.state.win_reason)

    def test_uncover_trap_mechanism(self):
        """
        Lifting a piece that reveals an opponent's 4-in-a-row:
        If active player fails to block, opponent wins immediately!
        """
        # Set up a board where Orange has 3 visible in row 0: (0,0), (0,1), (0,2)
        # And at (0,3), Orange has a piece covered by a Blue Large piece!
        self.state.board[0][0] = [Piece(PLAYER_ORANGE, 1)]
        self.state.board[0][1] = [Piece(PLAYER_ORANGE, 1)]
        self.state.board[0][2] = [Piece(PLAYER_ORANGE, 1)]
        self.state.board[0][3] = [Piece(PLAYER_ORANGE, 1), Piece(PLAYER_BLUE, 3)]

        # It is Blue's turn
        self.state.current_player = PLAYER_BLUE

        # If Blue lifts from (0,3), Orange's 4-in-a-row (0,0..0,3) is revealed!
        # If Blue moves to (4, 4), row 0 is NOT blocked -> Orange should win via Uncover Trap!
        bad_move = Move(PLAYER_BLUE, 'move', from_pos=(0, 3), to_pos=(4, 4))
        self.state.apply_move(bad_move)

        self.assertTrue(self.state.game_over)
        self.assertEqual(self.state.winner, PLAYER_ORANGE)
        self.assertIn("Uncover Trap", self.state.win_reason)

    def test_forgetting_memory_belief(self):
        """
        Test that moves older than memory_window (5) have their buried pieces forgotten.
        """
        # Blue places small piece at (2, 2) at turn 1
        m1 = Move(PLAYER_BLUE, 'drop', (2, 2), size=1)
        self.state.apply_move(m1)

        # Orange covers it with medium piece at turn 2
        m2 = Move(PLAYER_ORANGE, 'drop', (2, 2), size=2)
        self.state.apply_move(m2)

        # Now simulate turns up to turn 8 (m2 was covered at turn 2, 8 - 2 = 6 turns ago > 5)
        # Advance turns
        self.state.turn_number = 8

        # Check memory inspector
        snapshots = MemoryInspector.get_memory_snapshot(self.state, memory_window=5)
        self.assertEqual(len(snapshots), 1)
        record = snapshots[0]
        self.assertEqual(record['pos'], (2, 2))
        self.assertFalse(record['is_remembered'])
        self.assertIn("Forgotten", record['status_text'])

        # Belief state should NOT have the buried piece
        belief = build_belief_state(self.state, memory_window=5)
        self.assertEqual(len(belief.board[2][2]), 1)  # Only top piece remains!

    def test_ai_agent_decision(self):
        """AI agent should find a valid legal move and evaluate it."""
        agent = MinimaxAgent(player=PLAYER_ORANGE, max_depth=2, memory_window=5)
        agent.step_delay = 0  # Fast execution for tests

        # If Blue has 3 large pieces (size 3) at row 2, they cannot be gobbled.
        # Orange must block by dropping or moving to (2, 3).
        self.state.board[2][0] = [Piece(PLAYER_BLUE, 3)]
        self.state.board[2][1] = [Piece(PLAYER_BLUE, 3)]
        self.state.board[2][2] = [Piece(PLAYER_BLUE, 3)]
        self.state.current_player = PLAYER_ORANGE

        best_move = agent.get_action(self.state)
        self.assertIsNotNone(best_move)
        self.assertEqual(best_move.to_pos, (2, 3))

    def test_winning_lines_count_5(self):
        """On a 5x5 board, 5-in-a-row should have exactly 12 winning lines (5 horiz, 5 vert, 2 diag)."""
        lines_5 = generate_winning_lines(5)
        self.assertEqual(len(lines_5), 12)

    def test_win_streak_5_incomplete(self):
        """In a 5-in-a-row game, placing 4 pieces in a row must NOT trigger a win."""
        state_5 = GameState(win_streak=5)
        for c in range(4):
            state_5.board[0][c] = [Piece(PLAYER_BLUE, 2)]

        winner, line = state_5.check_winner_on_board()
        self.assertIsNone(winner)
        self.assertIsNone(line)

    def test_win_streak_5_horizontal(self):
        """In a 5-in-a-row game, completing 5 in a row horizontally triggers a win."""
        state_5 = GameState(win_streak=5)
        # Blue drops at (0, 0), (0, 1), (0, 2), (0, 3)
        for c in range(4):
            state_5.board[0][c] = [Piece(PLAYER_BLUE, 2)]
        state_5.current_player = PLAYER_BLUE

        # Drop the 5th piece at (0, 4)
        m = Move(PLAYER_BLUE, 'drop', (0, 4), size=2)
        state_5.apply_move(m)

        self.assertTrue(state_5.game_over)
        self.assertEqual(state_5.winner, PLAYER_BLUE)
        self.assertIn("5-in-a-row", state_5.win_reason)

    def test_win_streak_5_diagonal(self):
        """In a 5-in-a-row game, 5 pieces on main diagonal wins."""
        state_5 = GameState(win_streak=5)
        for i in range(5):
            state_5.board[i][i] = [Piece(PLAYER_ORANGE, 1)]

        winner, line = state_5.check_winner_on_board()
        self.assertEqual(winner, PLAYER_ORANGE)
        self.assertEqual(line, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])

    def test_win_streak_5_uncover_trap(self):
        """In 5-in-a-row game, lifting a piece that uncovers an opponent's 5-in-a-row triggers uncover trap."""
        state_5 = GameState(win_streak=5)
        # Orange has pieces along row 0 from (0,0) to (0,3)
        for c in range(4):
            state_5.board[0][c] = [Piece(PLAYER_ORANGE, 1)]
        # At (0,4), Orange has piece covered by Blue Large piece (size 3)
        state_5.board[0][4] = [Piece(PLAYER_ORANGE, 1), Piece(PLAYER_BLUE, 3)]
        state_5.current_player = PLAYER_BLUE

        # Blue lifts piece at (0,4) and moves to (4,4) leaving row 0 unblocked
        bad_move = Move(PLAYER_BLUE, 'move', from_pos=(0, 4), to_pos=(4, 4))
        state_5.apply_move(bad_move)

        self.assertTrue(state_5.game_over)
        self.assertEqual(state_5.winner, PLAYER_ORANGE)
        self.assertIn("Uncover Trap", state_5.win_reason)

    def test_turn_order_enforced(self):
        """apply_move must reject moves played out of turn."""
        self.assertEqual(self.state.current_player, PLAYER_BLUE)
        # Orange attempts to move on Blue's turn
        illegal_orange_move = Move(PLAYER_ORANGE, 'drop', (0, 0), size=1)
        self.assertFalse(self.state.apply_move(illegal_orange_move))
        # Board remains empty and turn does not advance
        self.assertEqual(self.state.current_player, PLAYER_BLUE)
        self.assertEqual(self.state.turn_number, 1)

    def test_precomputed_winning_lines(self):
        """Precomputed WINNING_LINES_4 and WINNING_LINES_5 should match expected counts and types."""
        from game_engine import WINNING_LINES_4, WINNING_LINES_5
        self.assertEqual(len(WINNING_LINES_4), 28)
        self.assertEqual(len(WINNING_LINES_5), 12)
        # Check that clones reuse the precomputed winning lines list
        clone = self.state.clone()
        self.assertEqual(clone.winning_lines, self.state.winning_lines)

    def test_default_win_streak_is_5(self):
        """GameState() default constructor must initialize with win_streak=5 and 12 winning lines."""
        default_state = GameState()
        self.assertEqual(default_state.win_streak, 5)
        self.assertEqual(len(default_state.winning_lines), 12)

    def test_ai_agent_decision_5_in_a_row_threat(self):
        """AI agent in 5-in-a-row mode prioritizes blocking 4-of-5 threat."""
        state = GameState(win_streak=5)
        agent = MinimaxAgent(player=PLAYER_ORANGE, max_depth=2, memory_window=5)
        agent.step_delay = 0

        # Human (Blue) has 4 large pieces across row 0: (0,0), (0,1), (0,2), (0,3).
        # (0,4) is empty. Orange MUST drop or move to (0,4) to block human from winning!
        for c in range(4):
            state.board[0][c] = [Piece(PLAYER_BLUE, 3)]
        state.current_player = PLAYER_ORANGE

        best_move = agent.get_action(state)
        self.assertIsNotNone(best_move)
        self.assertEqual(best_move.to_pos, (0, 4))

    def test_ai_agent_5_in_a_row_gobble_threat(self):
        """AI recognizes that an opponent 4-of-5 threat blocked only by a small AI piece can be gobbled and secures it."""
        state = GameState(win_streak=5)
        # Human has 4 medium pieces at row 0: (0,0)..(0,3)
        for c in range(4):
            state.board[0][c] = [Piece(PLAYER_BLUE, 2)]
        # AI has small piece at (0,4) which Human can gobble with Large piece
        state.board[0][4] = [Piece(PLAYER_ORANGE, 1)]
        state.current_player = PLAYER_ORANGE

        agent = MinimaxAgent(player=PLAYER_ORANGE, max_depth=1, memory_window=5)
        agent.step_delay = 0
        best_move = agent.get_action(state)
        self.assertIsNotNone(best_move)
        # AI must reinforce (0,4) by dropping a Large piece or moving to secure
        self.assertEqual(best_move.to_pos, (0, 4))

    def test_ai_agent_5_in_a_row_gobble_win(self):
        """AI with 4 pieces in a line gobbles opponent's smaller piece on the 5th cell to win."""
        state = GameState(win_streak=5)
        # AI has 4 pieces at row 1: (1,0)..(1,3)
        for c in range(4):
            state.board[1][c] = [Piece(PLAYER_ORANGE, 2)]
        # Human has a small piece at (1,4)
        state.board[1][4] = [Piece(PLAYER_BLUE, 1)]
        state.current_player = PLAYER_ORANGE

        agent = MinimaxAgent(player=PLAYER_ORANGE, max_depth=1, memory_window=5)
        agent.step_delay = 0
        best_move = agent.get_action(state)
        self.assertIsNotNone(best_move)
        # AI must gobble (1,4) to achieve 5-in-a-row!
        self.assertEqual(best_move.to_pos, (1, 4))


if __name__ == '__main__':
    unittest.main()
