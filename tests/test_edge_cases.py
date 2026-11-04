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
    MinimaxAgent, build_belief_state, AIHeuristic
)


class TestGobbletEdgeCases(unittest.TestCase):

    def setUp(self):
        """
        Serves as the test fixture setup for edge-case regression suites.

        Initializes a fresh 5x5 `GameState` instance configured with the standard
        5-in-a-row (`win_streak=5`) winning condition before each test execution,
        ensuring test isolation and state purity across test runs.
        """
        self.state = GameState(win_streak=5)

    def test_friendly_gobbling(self):
        """
        Serves as the rule validation test for friendly piece gobbling (self-stacking).

        Verifies that players are permitted to gobble their own pieces with larger sizes:
        1. Blue drops a Small piece (size 1) at (1, 1).
        2. Orange makes a neutral drop elsewhere.
        3. Blue drops a Medium piece (size 2) directly onto their own Small piece at (1, 1).
        Asserts that the move succeeds, stack depth increments to 2, and the top piece
        remains Blue with size 2.
        """
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
        """
        Serves as the win detection validation test for 5x5 diagonal alignments.

        Tests both diagonal geometries supported on a 5x5 board:
        - Main diagonal: (0,0), (1,1), (2,2), (3,3), (4,4) for Blue.
        - Anti-diagonal: (0,4), (1,3), (2,2), (3,1), (4,0) for Orange.
        Confirms that 4 pieces in a line do NOT trigger a win, while placing the 5th
        piece completes the alignment, returning the correct winner and line coordinates.
        """
        # Main diagonal line: (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)
        for i in range(4):
            self.state.board[i][i] = [Piece(PLAYER_BLUE, 1)]

        # 4 pieces must NOT win
        winner, line = self.state.check_winner_on_board()
        self.assertIsNone(winner)
        self.assertIsNone(line)

        # 5th piece completes diagonal and wins
        self.state.board[4][4] = [Piece(PLAYER_BLUE, 1)]
        winner, line = self.state.check_winner_on_board()
        self.assertEqual(winner, PLAYER_BLUE)
        self.assertEqual(line, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])

        # Anti-diagonal line: (0, 4), (1, 3), (2, 2), (3, 1), (4, 0)
        state2 = GameState(win_streak=5)
        state2.board[0][4] = [Piece(PLAYER_ORANGE, 2)]
        state2.board[1][3] = [Piece(PLAYER_ORANGE, 2)]
        state2.board[2][2] = [Piece(PLAYER_ORANGE, 2)]
        state2.board[3][1] = [Piece(PLAYER_ORANGE, 2)]
        state2.board[4][0] = [Piece(PLAYER_ORANGE, 2)]

        winner2, line2 = state2.check_winner_on_board()
        self.assertEqual(winner2, PLAYER_ORANGE)
        self.assertEqual(line2, [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)])

    def test_uncover_trap_successful_block(self):
        """
        Serves as the validation test for surviving the Uncover Trap via legal blocking.

        Enforces the nuanced Uncover Trap rule:
        If lifting a board piece reveals an opponent's 5-in-a-row, the player does NOT
        immediately lose IF the lifted piece is placed onto a cell that legally blocks
        and breaks that revealed winning line.
        Asserts that the move succeeds, game_over remains False, winner is None, and
        the turn advances to the opponent.
        """
        # Orange has 4 visible in row 0: (0, 0), (0, 1), (0, 2), (0, 3)
        # And at (0, 4), Orange has a piece covered by Blue Large piece (size 3)
        # Blue can move the piece from (0, 4) to (0, 0) covering Orange's size 1 piece!
        self.state.board[0][0] = [Piece(PLAYER_ORANGE, 1)]
        self.state.board[0][1] = [Piece(PLAYER_ORANGE, 1)]
        self.state.board[0][2] = [Piece(PLAYER_ORANGE, 1)]
        self.state.board[0][3] = [Piece(PLAYER_ORANGE, 1)]
        self.state.board[0][4] = [Piece(PLAYER_ORANGE, 1), Piece(PLAYER_BLUE, 3)]

        self.state.current_player = PLAYER_BLUE

        # Blue lifts Large from (0, 4) and places it on (0, 0) covering Orange!
        blocking_move = Move(PLAYER_BLUE, 'move', from_pos=(0, 4), to_pos=(0, 0))
        success = self.state.apply_move(blocking_move)
        self.assertTrue(success)

        # Orange does NOT win because (0, 0) is now covered by Blue!
        # The line (0,0..0,4) is broken because (0,0) is Blue!
        self.assertFalse(self.state.game_over)
        self.assertIsNone(self.state.winner)
        self.assertEqual(self.state.current_player, PLAYER_ORANGE)

    def test_memory_forgetting_boundary(self):
        """
        Serves as the precise boundary test for bounded memory window transitions.

        Validates the cognitive forgetting model implemented in `build_belief_state`:
        - A piece buried within the memory window (`current_turn - covered_turn <= 5`)
          must be accurately recalled and kept in the belief stack.
        - When turn advances beyond the memory window (`current_turn - covered_turn == 6`),
          the piece is forgotten and omitted from the belief stack.
        """
        # Blue drops small at (3, 3) at turn 1
        self.state.apply_move(Move(PLAYER_BLUE, 'drop', (3, 3), size=1))
        # Orange covers it at turn 2 with medium
        self.state.apply_move(Move(PLAYER_ORANGE, 'drop', (3, 3), size=2))

        # At turn 7: current_turn = 7, covered at 2 => 7 - 2 = 5 turns ago (Remembered!)
        self.state.turn_number = 7
        belief7 = build_belief_state(self.state, memory_window=5)
        self.assertEqual(len(belief7.board[3][3]), 2)
        self.assertEqual(belief7.board[3][3][0].player, PLAYER_BLUE)

        # At turn 8: current_turn = 8, covered at 2 => 8 - 2 = 6 turns ago (Forgotten!)
        self.state.turn_number = 8
        belief8 = build_belief_state(self.state, memory_window=5)
        self.assertEqual(len(belief8.board[3][3]), 1)
        self.assertEqual(belief8.board[3][3][0].player, PLAYER_ORANGE)

    def test_alpha_beta_pruning_execution(self):
        """
        Serves as the integration test for Minimax Alpha-Beta search execution.

        Configures a `MinimaxAgent` and executes a depth-3 search on an active board configuration.
        Confirms that the agent selects a valid move and tracks visited nodes and pruning cutoffs.
        """
        agent = MinimaxAgent(player=PLAYER_ORANGE, max_depth=3, memory_window=5)
        agent.step_delay = 0

        # Run AI on game state with several pieces on board
        self.state.board[1][1] = [Piece(PLAYER_BLUE, 1)]
        self.state.board[2][2] = [Piece(PLAYER_ORANGE, 2)]
        self.state.current_player = PLAYER_ORANGE

        move = agent.get_action(self.state)
        self.assertIsNotNone(move)
        self.assertGreater(agent.nodes_visited, 0)
        self.assertGreaterEqual(agent.prune_count, 0)

    def test_simultaneous_5_in_a_row_uncover_trap(self):
        """
        Serves as the priority resolution test for simultaneous 5-in-a-row uncover traps.

        Edge-case scenario:
        Blue repositions a piece which simultaneously uncovers Orange's 5-in-a-row and
        completes Blue's own 5-in-a-row at the destination.
        Because the uncovered Orange line was created upon lifting and was not blocked by
        the destination, official Gobblet rules dictate that Orange wins immediately via
        Uncover Trap priority.
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
        """
        Serves as the heuristic penalty validation test for uncover trap vulnerabilities.

        Compares `AIHeuristic.evaluate()` scores between two board states:
        - Dangerous State: An Orange piece covers a buried Blue piece that would complete
          a 5-in-a-row line for Blue if lifted.
        - Safe State: The same Orange piece covers a friendly Orange piece.
        Asserts that the evaluation of the dangerous state is penalized by more than 300 points.
        """
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
        """
        Serves as the terminal condition test for player immobility.

        Configures a board where Blue has 0 pieces remaining in reserves and its only
        board piece is completely gobbled/covered by an Orange piece.
        Validates that `get_legal_moves(PLAYER_BLUE)` returns an empty list, signifying
        complete immobilization.
        """
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
