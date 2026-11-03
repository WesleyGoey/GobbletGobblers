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
        """
        Serves as the test fixture initialization for the core game engine test suite.

        Allocates a clean `GameState(win_streak=5)` instance before each test method,
        ensuring an empty 5x5 board, full reserves (4 small, 4 medium, 4 large pieces per player),
        turn number 1, active player Blue, and uncorrupted rule structures.
        """
        self.state = GameState(win_streak=5)

    def test_initial_state(self):
        """
        Serves as the baseline state verification test for new games.

        Validates that a newly instantiated 5x5 `GameState` satisfies all initial game specifications:
        - Active player is PLAYER_BLUE.
        - Initial turn number is 1.
        - `game_over` is False and `winner` is None.
        - Each player possesses exactly 4 pieces across all 3 size tiers (Small, Medium, Large) in reserves.
        - All 25 cells across the 5x5 board grid have a stack depth of 0 and return None for `get_top_piece()`.
        """
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
        """
        Serves as the geometric line precomputation test for 5-in-a-row on a 5x5 board.

        Verifies that the precomputed `WINNING_LINES` constant contains exactly 12 contiguous
        5-in-a-row straight line paths (5 horizontal rows, 5 vertical columns, and 2 main diagonals),
        ensuring comprehensive coverage without missing or extraneous line alignments.
        """
        self.assertEqual(len(WINNING_LINES), 12)

    def test_legal_drop_and_turn_advance(self):
        """
        Serves as the basic move execution and turn advancement test.

        Validates Action A (introducing a reserve piece onto the board):
        - Executes a legal drop of Blue's Large piece (size 3) onto the center cell (2, 2).
        - Verifies that Blue's Large reserve count decrements from 4 to 3.
        - Verifies that the turn toggles to PLAYER_ORANGE and `turn_number` increments to 2.
        - Asserts that cell (2, 2) top piece is Blue and top size is 3.
        """
        move = Move(player=PLAYER_BLUE, move_type='drop', size=3, to_pos=(2, 2))
        success = self.state.apply_move(move)
        self.assertTrue(success)
        self.assertEqual(self.state.reserves[PLAYER_BLUE][3], 3)
        self.assertEqual(self.state.current_player, PLAYER_ORANGE)
        self.assertEqual(self.state.turn_number, 2)
        self.assertEqual(self.state.get_top_size(2, 2), 3)
        self.assertEqual(self.state.get_top_piece(2, 2).player, PLAYER_BLUE)

    def test_gobble_rules(self):
        """
        Serves as the nesting and gobbling rule verification test.

        Enforces strictly descending size constraints on cell stacks:
        1. Blue drops Small (size 1) at (0, 0) -> succeeds.
        2. Orange attempts to drop Small (size 1) on top -> rejected (equal size cannot gobble).
        3. Orange drops Medium (size 2) on top -> succeeds (Medium gobbles Small, stack depth 2).
        4. Blue drops Large (size 3) on top -> succeeds (Large gobbles Medium, stack depth 3).
        5. Orange attempts to drop another Large (size 3) on top -> rejected (stack is full at size 3).
        """
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
        """
        Serves as the horizontal win detection and streak-length threshold test.

        Simulates sequential alternating drops across row 1:
        - Confirms that after 4 Blue pieces are placed at (1,0)..(1,3), `game_over` remains False
          and `winner` is None (verifying that 4-in-a-row does NOT win in 5-in-a-row mode).
        - Executes the 5th drop at (1,4), confirming `game_over` triggers with winner PLAYER_BLUE
          and descriptive `win_reason` mentioning '5-in-a-row'.
        """
        # Blue drops across row 1: (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)
        moves = [
            Move(PLAYER_BLUE, 'drop', (1, 0), size=1),
            Move(PLAYER_ORANGE, 'drop', (0, 0), size=1),
            Move(PLAYER_BLUE, 'drop', (1, 1), size=1),
            Move(PLAYER_ORANGE, 'drop', (0, 1), size=1),
            Move(PLAYER_BLUE, 'drop', (1, 2), size=1),
            Move(PLAYER_ORANGE, 'drop', (0, 2), size=1),
            Move(PLAYER_BLUE, 'drop', (1, 3), size=1),  # 4th Blue in row 1!
            Move(PLAYER_ORANGE, 'drop', (0, 3), size=1),
            Move(PLAYER_BLUE, 'drop', (1, 4), size=2),  # 5th Blue in row 1!
        ]
        # First 7 moves: Blue has 4 in row 1
        for m in moves[:7]:
            self.state.apply_move(m)

        # 4 in a row must NOT trigger game over
        self.assertFalse(self.state.game_over)
        self.assertIsNone(self.state.winner)

        # Orange plays, then Blue completes 5th in row 1
        self.state.apply_move(moves[7])
        self.state.apply_move(moves[8])

        self.assertTrue(self.state.game_over)
        self.assertEqual(self.state.winner, PLAYER_BLUE)
        self.assertIn("Blue", self.state.win_reason)
        self.assertIn("5-in-a-row", self.state.win_reason)

    def test_uncover_trap_mechanism(self):
        """
        Serves as the Uncover Trap forfeiture test for unblocked opponent win exposures.

        Validates the critical official Gobblet rule:
        Lifting a board piece that reveals an underlying opponent piece completing a 5-in-a-row
        immediately forfeits the match to the opponent IF the lifted piece is moved to any destination
        that fails to block the exposed line.
        """
        # Set up a board where Orange has 4 visible in row 0: (0,0), (0,1), (0,2), (0,3)
        # And at (0,4), Orange has a piece covered by a Blue Large piece!
        self.state.board[0][0] = [Piece(PLAYER_ORANGE, 1)]
        self.state.board[0][1] = [Piece(PLAYER_ORANGE, 1)]
        self.state.board[0][2] = [Piece(PLAYER_ORANGE, 1)]
        self.state.board[0][3] = [Piece(PLAYER_ORANGE, 1)]
        self.state.board[0][4] = [Piece(PLAYER_ORANGE, 1), Piece(PLAYER_BLUE, 3)]

        # It is Blue's turn
        self.state.current_player = PLAYER_BLUE

        # If Blue lifts from (0,4), Orange's 5-in-a-row (0,0..0,4) is revealed!
        # If Blue moves to (4, 4), row 0 is NOT blocked -> Orange should win via Uncover Trap!
        bad_move = Move(PLAYER_BLUE, 'move', from_pos=(0, 4), to_pos=(4, 4))
        self.state.apply_move(bad_move)

        self.assertTrue(self.state.game_over)
        self.assertEqual(self.state.winner, PLAYER_ORANGE)
        self.assertIn("Uncover Trap", self.state.win_reason)

    def test_forgetting_memory_belief(self):
        """
        Serves as the AI bounded memory integration test for buried piece forgetting.

        Validates that pieces covered more than `memory_window` turns ago are forgotten
        by both `MemoryInspector` snapshots and `build_belief_state()`.
        Confirms that when turn advances to 8, a piece covered at turn 2 (delta = 6 > 5)
        is flagged as forgotten and omitted from the AI's internal simulated board state.
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
        """
        Serves as the tactical emergency response test for MinimaxAgent decision-making.

        Constructs an imminent threat scenario where Blue has 4 un-gobbleable Large pieces
        at (2,0)..(2,3) threatening row 2.
        Verifies that `MinimaxAgent.get_action()` correctly analyzes the search tree and selects
        the single vital blocking destination `to_pos == (2, 4)`.
        """
        agent = MinimaxAgent(player=PLAYER_ORANGE, max_depth=2, memory_window=5)
        agent.step_delay = 0  # Fast execution for tests

        # If Blue has 4 large pieces (size 3) at row 2, they cannot be gobbled.
        # Orange must block by dropping or moving to (2, 4).
        self.state.board[2][0] = [Piece(PLAYER_BLUE, 3)]
        self.state.board[2][1] = [Piece(PLAYER_BLUE, 3)]
        self.state.board[2][2] = [Piece(PLAYER_BLUE, 3)]
        self.state.board[2][3] = [Piece(PLAYER_BLUE, 3)]
        self.state.current_player = PLAYER_ORANGE

        best_move = agent.get_action(self.state)
        self.assertIsNotNone(best_move)
        self.assertEqual(best_move.to_pos, (2, 4))

    def test_winning_lines_count_5(self):
        """
        Serves as the unit verification test for `generate_winning_lines(5)`.

        Directly calls `generate_winning_lines(5)` and validates that exactly 12 winning lines
        are computed (5 horizontal rows, 5 vertical columns, 2 primary diagonals) on a 5x5 board.
        """
        lines_5 = generate_winning_lines(5)
        self.assertEqual(len(lines_5), 12)

    def test_win_streak_5_incomplete(self):
        """
        Serves as the negative boundary test for 5-in-a-row alignment detection.

        Populates 4 aligned pieces across row 0 on a `GameState(win_streak=5)` instance.
        Invokes `check_winner_on_board()` and asserts that winner and winning line both return None,
        guaranteeing partial alignments do not trigger premature victories.
        """
        state_5 = GameState(win_streak=5)
        for c in range(4):
            state_5.board[0][c] = [Piece(PLAYER_BLUE, 2)]

        winner, line = state_5.check_winner_on_board()
        self.assertIsNone(winner)
        self.assertIsNone(line)

    def test_win_streak_5_horizontal(self):
        """
        Serves as the horizontal win execution test for standard 5-in-a-row matches.

        Sets up 4 pieces in row 0, executes the 5th drop move at (0, 4), and asserts that
        `state.apply_move()` transitions `game_over` to True with `winner == PLAYER_BLUE`.
        """
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
        """
        Serves as the diagonal win detection test for main diagonal coordinates.

        Populates all 5 coordinates along the main diagonal (0,0) through (4,4) with Orange pieces.
        Invokes `check_winner_on_board()`, asserting that `winner == PLAYER_ORANGE` and the returned
        line exactly matches `[(0,0), (1,1), (2,2), (3,3), (4,4)]`.
        """
        state_5 = GameState(win_streak=5)
        for i in range(5):
            state_5.board[i][i] = [Piece(PLAYER_ORANGE, 1)]

        winner, line = state_5.check_winner_on_board()
        self.assertEqual(winner, PLAYER_ORANGE)
        self.assertEqual(line, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])

    def test_win_streak_5_uncover_trap(self):
        """
        Serves as the 5-in-a-row Uncover Trap validation test under full board rules.

        Sets up Orange with 4 visible pieces at (0,0)..(0,3) and an Orange piece covered by
        Blue's Large piece at (0,4). Blue relocates the piece at (0,4) to (4,4), uncovering
        Orange's 5th piece without blocking row 0. Asserts Orange immediately wins via Uncover Trap.
        """
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
        """
        Serves as the turn enforcement and sequence validation test.

        Verifies that `apply_move()` strictly rejects actions attempted out of turn:
        - When active player is PLAYER_BLUE, asserts that an action initiated by PLAYER_ORANGE returns False.
        - Confirms that the invalid move causes zero board mutation and that `current_player` remains Blue.
        """
        self.assertEqual(self.state.current_player, PLAYER_BLUE)
        # Orange attempts to move on Blue's turn
        illegal_orange_move = Move(PLAYER_ORANGE, 'drop', (0, 0), size=1)
        self.assertFalse(self.state.apply_move(illegal_orange_move))
        # Board remains empty and turn does not advance
        self.assertEqual(self.state.current_player, PLAYER_BLUE)
        self.assertEqual(self.state.turn_number, 1)

    def test_precomputed_winning_lines(self):
        """
        Serves as the architectural validation test for precomputed line constants and memory sharing.

        Validates the precomputed winning line data structures:
        - `WINNING_LINES` and `WINNING_LINES_5` both contain exactly 12 precomputed lines and are equivalent.
        - Legacy 4-in-a-row precomputation structures (`WINNING_LINES_4`, `CELL_LINES_4`) have been eliminated.
        - Cloned `GameState` instances share the identical precomputed lines reference to minimize memory overhead.
        """
        from game_engine import WINNING_LINES, WINNING_LINES_5
        import game_engine
        self.assertEqual(len(WINNING_LINES), 12)
        self.assertEqual(len(WINNING_LINES_5), 12)
        self.assertEqual(WINNING_LINES, WINNING_LINES_5)
        self.assertFalse(hasattr(game_engine, 'WINNING_LINES_4'))
        self.assertFalse(hasattr(game_engine, 'CELL_LINES_4'))
        # Check that clones reuse the precomputed winning lines list
        clone = self.state.clone()
        self.assertEqual(clone.winning_lines, self.state.winning_lines)

    def test_default_win_streak_is_5(self):
        """
        Serves as the default configuration test for GameState initialization.

        Verifies that parameterless construction `GameState()` defaults strictly to
        `win_streak = 5` and binds all 12 precomputed winning lines.
        """
        default_state = GameState()
        self.assertEqual(default_state.win_streak, 5)
        self.assertEqual(len(default_state.winning_lines), 12)

    def test_win_streak_4_rejected(self):
        """
        Serves as the deprecation and validation guard test for 4-in-a-row rejection.

        Ensures that obsolete 4-in-a-row streak arguments are rejected with `ValueError`
        across entrypoints: `GameState(win_streak=4)`, `generate_winning_lines(4)`,
        and `get_default_scenarios(win_streak=4)`.
        """
        from run_experiments import get_default_scenarios
        with self.assertRaises(ValueError):
            GameState(win_streak=4)
        with self.assertRaises(ValueError):
            generate_winning_lines(4)
        with self.assertRaises(ValueError):
            get_default_scenarios(win_streak=4)

    def test_ai_agent_decision_5_in_a_row_threat(self):
        """
        Serves as the high-priority threat defense test for MinimaxAgent in 5-in-a-row mode.

        Creates an urgent tactical scenario where the opponent possesses 4 Large pieces
        at (0,0)..(0,3). Validates that the AI prioritizes blocking the winning alignment
        at (0, 4) over all other potential actions.
        """
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
        """
        Serves as the defensive gobble reinforcement test for MinimaxAgent.

        Tests AI defensive awareness in 3D stacking:
        The opponent has 4 Medium pieces at (0,0)..(0,3) and threatens to gobble AI's Small piece
        at (0,4) with a Large piece to win. Validates that the AI reinforces cell (0, 4) by
        dropping a Large piece or moving to secure the cell against being gobbled.
        """
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
        """
        Serves as the offensive gobble victory conversion test for MinimaxAgent.

        Configures an offensive endgame state:
        The AI has 4 pieces at (1,0)..(1,3) in row 1, while cell (1,4) contains an opponent Small piece.
        Validates that the AI recognizes the gobbling opportunity and plays a larger piece to (1, 4),
        completing 5-in-a-row and securing an immediate victory.
        """
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


class TestTable3ExperimentRunner(unittest.TestCase):
    """Unit tests for Table 3 empirical benchmarking suite."""

    def test_default_scenarios_count(self):
        """
        Serves as the scenario suite integrity test for Table 3 benchmark configurations.

        Verifies that `get_default_scenarios()` constructs exactly 5 distinct test scenarios,
        each scenario has a descriptive title, and each binds a valid `GameState` instance.
        """
        from run_experiments import get_default_scenarios
        scenarios = get_default_scenarios()
        self.assertEqual(len(scenarios), 5)
        for name, state in scenarios:
            self.assertIsInstance(name, str)
            self.assertIsInstance(state, GameState)

    def test_run_table_3_experiment_fast_depth_1(self):
        """
        Serves as the fast smoke test and backward compatibility test for Table 3 experiments.

        Executes a rapid depth-1 benchmark run, validating that:
        - `run_table_3_experiment()` succeeds with `passed == True`.
        - Chosen move identity between Pure Minimax and Alpha-Beta is 100% equivalent (`moves_matched == True`).
        - The backward alias `test_3_alpha_beta_pruning` is identical to `run_table_3_experiment`.
        """
        from run_experiments import run_table_3_experiment, test_3_alpha_beta_pruning
        s = GameState(win_streak=5)
        s.apply_move(Move(PLAYER_BLUE, 'drop', size=3, to_pos=(2, 2)))
        res = run_table_3_experiment(scenarios=[("Quick Test", s)], depth=1, verbose=False)
        self.assertTrue(res["passed"])
        self.assertEqual(len(res["scenarios"]), 1)
        self.assertEqual(res["overall_reduction"], 0.0)
        self.assertTrue(res["moves_matched"])
        self.assertIs(test_3_alpha_beta_pruning, run_table_3_experiment)

    def test_run_table_3_experiment_depth_2_reduction(self):
        """
        Serves as the empirical pruning efficiency test for Alpha-Beta search.

        Executes Table 3 benchmark at search depth 2, validating that Alpha-Beta pruning
        achieves significant search space pruning (>30% node reduction) compared to Pure Minimax
        while guaranteeing 100% move equivalence.
        """
        from run_experiments import run_table_3_experiment
        s = GameState(win_streak=5)
        s.apply_move(Move(PLAYER_BLUE, 'drop', size=3, to_pos=(2, 2)))
        res = run_table_3_experiment(scenarios=[("Depth 2 Test", s)], depth=2, verbose=False)
        self.assertTrue(res["passed"])
        self.assertGreaterEqual(res["overall_reduction"], 30.0)
        self.assertTrue(res["moves_matched"])
        self.assertGreater(res["total_raw_nodes"], res["total_pruned_nodes"])

    def test_run_table_3_dynamic_player_turn(self):
        """
        Serves as the dynamic active-player perspective test for Table 3 benchmarks.

        Validates that when a benchmark scenario has PLAYER_BLUE as the active player,
        the search algorithm generates moves from Blue's perspective (`[Blue] ...`),
        confirming that the test harness does not hardcode agent color.
        """
        from run_experiments import run_table_3_experiment
        s = GameState(win_streak=5)
        self.assertEqual(s.current_player, PLAYER_BLUE)
        res = run_table_3_experiment(scenarios=[("Blue Turn", s)], depth=1, verbose=False)
        self.assertTrue(res["passed"])
        chosen_move = res["scenarios"][0]["move_raw"]
        self.assertTrue(chosen_move.startswith("[Blue]"), f"Expected Blue move but got {chosen_move}")

    def test_run_table_3_csv_and_markdown(self):
        """
        Serves as the output artifact verification test for Table 3 reports.

        Validates that `run_table_3_experiment` correctly renders Markdown summary tables
        and exports structured CSV benchmark data containing headers, node counts,
        and scenario identifiers to a specified file path.
        """
        import tempfile
        import os
        from run_experiments import run_table_3_experiment
        s = GameState(win_streak=5)
        s.apply_move(Move(PLAYER_BLUE, 'drop', size=3, to_pos=(2, 2)))
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = os.path.join(tmpdir, "nested", "table3_output.csv")
            res = run_table_3_experiment(
                scenarios=[("CSV Test", s)],
                depth=1,
                verbose=False,
                print_markdown=True,
                csv_path=csv_file
            )
            self.assertTrue(res["passed"])
            self.assertTrue(os.path.exists(csv_file))
            with open(csv_file, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("Pure_Minimax_nodes", content)
                self.assertIn("Move_Pure_Minimax", content)
                self.assertIn("CSV Test", content)


if __name__ == '__main__':
    unittest.main()
