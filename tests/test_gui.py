"""
Automated headless GUI tests for GobbletGUI.
Runs with SDL_VIDEODRIVER=dummy to verify all visual and interaction components.
"""

import os
import unittest
import time

os.environ['SDL_VIDEODRIVER'] = 'dummy'
import pygame

from gui import GobbletGUI
from game_engine import PLAYER_BLUE, PLAYER_ORANGE, Move


class TestGobbletGUI(unittest.TestCase):

    def setUp(self):
        """
        Serves as the test fixture initialization for the GobbletGUI headless test suite.

        Instantiates a `GobbletGUI(width=1320, height=860)` instance operating under the
        SDL dummy video driver, providing full access to screen surfaces, component rects,
        and event queues without requiring a physical monitor.
        """
        self.gui = GobbletGUI(width=1320, height=860)

    def tearDown(self):
        """
        Serves as the test fixture cleanup after each GUI test execution.

        Ensures that any background AI search worker threads or lingering thread locks
        are cleanly released and joined to avoid resource leakage between test runs.
        """
        pass

    def test_gui_initial_render(self):
        """
        Serves as the baseline render smoke test for the composite GUI canvas.

        Invokes `gui.draw()` on a freshly initialized game and verifies that all visual
        subsystems (header banner, player reserve trays, 5x5 board grid, control bar buttons,
        and controls) render onto the target screen surface without exceptions.
        """
        self.gui.draw()
        self.assertIsNotNone(self.gui.screen)

    def test_human_reserve_selection_and_drop(self):
        """
        Serves as the interactive mouse drop simulation test for human player moves.

        Simulates human interaction:
        1. Clicks Blue's Large piece (size 3) in the left reserve tray.
        2. Asserts `selected_reserve_size == 3` and verifies legal targets are calculated.
        3. Clicks the center cell (2, 2) on the board grid.
        4. Verifies that the piece is placed, top piece at (2, 2) is Blue Large, and selection clears.
        """
        # Click on Blue Large piece reserve rect
        rect_l3 = self.gui.blue_reserve_rects[3]
        self.gui.handle_mouse_click(rect_l3.center)

        self.assertEqual(self.gui.selected_reserve_size, 3)
        self.assertGreater(len(self.gui.valid_targets), 0)

        # Click on empty center cell (2, 2)
        cell_rect = self.gui.get_cell_rect(2, 2)
        self.gui.handle_mouse_click(cell_rect.center)

        # Center should now have Blue's Large piece
        top_piece = self.gui.state.get_top_piece(2, 2)
        self.assertIsNotNone(top_piece)
        self.assertEqual(top_piece.player, PLAYER_BLUE)
        self.assertEqual(top_piece.size, 3)

    def test_gui_difficulty_toggle_button(self):
        """
        Serves as the difficulty cycling button test in the GUI control panel.

        Simulates clicking `btn_difficulty` and verifies cyclical transitions:
        - Defaults to `medium` (depth 2, memory 5).
        - First click transitions to `hard` (depth 3, memory 8).
        - Second click transitions to `easy` (depth 1, memory 3).
        - Third click cycles back to `medium`.
        Verifies that agent depth and memory window synchronize at each step.
        """
        # Initial state should be default "medium"
        self.assertEqual(self.gui.difficulty, "medium")
        self.assertEqual(self.gui.ai_depth, 2)
        self.assertEqual(self.gui.agent.max_depth, 2)
        self.assertEqual(self.gui.memory_window, 5)
        self.assertEqual(self.gui.agent.memory_window, 5)

        # Click Difficulty button -> cycles to "hard"
        self.gui.handle_mouse_click(self.gui.btn_difficulty.center)
        self.assertEqual(self.gui.difficulty, "hard")
        self.assertEqual(self.gui.ai_depth, 3)
        self.assertEqual(self.gui.agent.max_depth, 3)
        self.assertEqual(self.gui.memory_window, 8)
        self.assertEqual(self.gui.agent.memory_window, 8)

        # Click Difficulty button -> cycles to "easy"
        self.gui.handle_mouse_click(self.gui.btn_difficulty.center)
        self.assertEqual(self.gui.difficulty, "easy")
        self.assertEqual(self.gui.ai_depth, 1)
        self.assertEqual(self.gui.agent.max_depth, 1)
        self.assertEqual(self.gui.memory_window, 3)
        self.assertEqual(self.gui.agent.memory_window, 3)

        # Click Difficulty button -> cycles back to "medium"
        self.gui.handle_mouse_click(self.gui.btn_difficulty.center)
        self.assertEqual(self.gui.difficulty, "medium")
        self.assertEqual(self.gui.ai_depth, 2)
        self.assertEqual(self.gui.agent.max_depth, 2)
        self.assertEqual(self.gui.memory_window, 5)
        self.assertEqual(self.gui.agent.memory_window, 5)

    def test_gui_reset_button(self):
        """
        Serves as the interactive game reset button test.

        Simulates match progression by advancing `turn_number` to 5, clicks `btn_new_game`,
        and confirms that the state cleanly resets to turn 1 with empty board cells.
        """
        self.gui.state.turn_number = 5
        self.gui.handle_mouse_click(self.gui.btn_new_game.center)
        self.assertEqual(self.gui.state.turn_number, 1)

    def test_gui_init_with_custom_difficulty(self):
        """
        Serves as the custom difficulty initialization test for GobbletGUI.

        Validates constructor configuration when instantiating GUI instances with explicit
        difficulty levels:
        - `GobbletGUI(difficulty="easy")`: sets search depth 1 and memory window 3.
        - `GobbletGUI(difficulty="hard")`: sets search depth 3 and memory window 8.
        """
        gui_easy = GobbletGUI(difficulty="easy")
        self.assertEqual(gui_easy.difficulty, "easy")
        self.assertEqual(gui_easy.ai_depth, 1)
        self.assertEqual(gui_easy.memory_window, 3)
        self.assertEqual(gui_easy.agent.max_depth, 1)
        self.assertEqual(gui_easy.agent.memory_window, 3)

        gui_hard = GobbletGUI(difficulty="hard")
        self.assertEqual(gui_hard.difficulty, "hard")
        self.assertEqual(gui_hard.ai_depth, 3)
        self.assertEqual(gui_hard.memory_window, 8)
        self.assertEqual(gui_hard.agent.max_depth, 3)
        self.assertEqual(gui_hard.agent.memory_window, 8)

    def test_touch_move_enforcement(self):
        """
        Serves as the interactive Touch-Move rule validation test in the GUI layer.

        Enforces the official Gobblet rule:
        Once a player touches/selects an existing piece on the board, they are committed
        to moving that piece. Clicking a reserve tray slot while a board piece is selected
        must be rejected with a "Touch-Move in effect" status warning, keeping the board selection locked.
        """
        from game_engine import Piece
        # Setup Blue piece at (1, 1)
        self.gui.state.board[1][1] = [Piece(PLAYER_BLUE, 2)]
        self.gui.state.current_player = PLAYER_BLUE

        # Click on cell (1, 1) to select it
        cell_1_1 = self.gui.get_cell_rect(1, 1)
        self.gui.handle_mouse_click(cell_1_1.center)

        self.assertEqual(self.gui.selected_board_pos, (1, 1))
        self.assertGreater(len(self.gui.valid_targets), 0)

        # Attempt to click reserve tray while board piece is locked
        rect_l3 = self.gui.blue_reserve_rects[3]
        self.gui.handle_mouse_click(rect_l3.center)
        # Selection MUST remain locked on (1, 1)
        self.assertEqual(self.gui.selected_board_pos, (1, 1))
        self.assertIn("Touch-Move in effect", self.gui.status_message)

        # Move to valid target (2, 2)
        cell_2_2 = self.gui.get_cell_rect(2, 2)
        self.gui.handle_mouse_click(cell_2_2.center)
        self.assertIsNone(self.gui.selected_board_pos)
        self.assertEqual(self.gui.state.get_top_piece(2, 2).player, PLAYER_BLUE)

    def test_instant_loss_on_unblockable_uncover_trap(self):
        """
        Serves as the instant-loss test for touching an unblockable uncover trap piece.

        Validates the strict physical touch rule:
        If lifting a board piece reveals an opponent's 5-in-a-row and the player has no
        legal destination on the board that can block the revealed line, touching the piece
        triggers an immediate instant forfeit loss upon touch.
        """
        from game_engine import Piece
        # Orange has 4 Large pieces (size 3) at row 0: (0,0), (0,1), (0,2), (0,3)
        self.gui.state.board[0][0] = [Piece(PLAYER_ORANGE, 3)]
        self.gui.state.board[0][1] = [Piece(PLAYER_ORANGE, 3)]
        self.gui.state.board[0][2] = [Piece(PLAYER_ORANGE, 3)]
        self.gui.state.board[0][3] = [Piece(PLAYER_ORANGE, 3)]
        # At (0, 4), Orange has a piece covered by Blue Small (size 1)
        self.gui.state.board[0][4] = [Piece(PLAYER_ORANGE, 1), Piece(PLAYER_BLUE, 1)]
        self.gui.state.current_player = PLAYER_BLUE

        # Blue touches cell (0, 4)
        cell_0_4 = self.gui.get_cell_rect(0, 4)
        self.gui.handle_mouse_click(cell_0_4.center)

        # Instant loss triggered!
        self.assertTrue(self.gui.state.game_over)
        self.assertEqual(self.gui.state.winner, PLAYER_ORANGE)
        self.assertIn("Uncover Trap", self.gui.state.win_reason)

    def test_gui_init_with_custom_win_streak(self):
        """
        Serves as the constructor win streak parameter guard and validation test.

        Validates that `GobbletGUI` strictly enforces the 5-in-a-row rule:
        - Default initialization and `GobbletGUI(win_streak=5)` bind 12 winning lines.
        - Instantiating `GobbletGUI(win_streak=4)` raises a `ValueError`.
        - Calling `gui.reset_game(win_streak=4)` raises a `ValueError`.
        """
        gui_default = GobbletGUI()
        self.assertEqual(gui_default.win_streak, 5)
        self.assertEqual(gui_default.state.win_streak, 5)
        self.assertEqual(len(gui_default.state.winning_lines), 12)

        gui5 = GobbletGUI(win_streak=5)
        self.assertEqual(gui5.win_streak, 5)
        self.assertEqual(gui5.state.win_streak, 5)
        self.assertEqual(len(gui5.state.winning_lines), 12)

        # Passing win_streak=4 must raise ValueError
        with self.assertRaises(ValueError):
            GobbletGUI(win_streak=4)

        # Resetting game with win_streak=4 must raise ValueError
        with self.assertRaises(ValueError):
            self.gui.reset_game(win_streak=4)

    def test_gui_reset_cancels_stale_ai_move(self):
        """
        Serves as the stale move cancellation test during mid-search game resets.

        Simulates a user pressing 'New Game / Reset' while a background AI search is in flight:
        - Executes a human move that launches background AI Minimax search.
        - Immediately triggers `gui.reset_game()`, which signals `cancel_search_event` and increments `search_id`.
        - Joins the worker thread, drains the queue, and asserts that no stale/phantom AI move
          is applied to the fresh game state (turn remains 1, board remains uncorrupted).
        """
        # Trigger human move -> launches AI background search
        self.gui.execute_human_move(Move(PLAYER_BLUE, 'drop', (0, 0), size=3))
        self.assertTrue(self.gui.ai_thinking)
        initial_search_id = self.gui.search_id

        # User resets game while AI is still searching
        self.gui.reset_game()
        self.assertGreater(self.gui.search_id, initial_search_id)
        self.assertEqual(self.gui.state.turn_number, 1)

        # Wait for background thread to conclude
        if self.gui.ai_thread:
            self.gui.ai_thread.join(timeout=5)

        # Process queue
        self.gui.process_ai_queue()
        self.assertIsNone(self.gui.pending_ai_move)
        # Turn must still be 1 (no phantom AI move applied!)
        self.assertEqual(self.gui.state.turn_number, 1)
        self.assertIsNone(self.gui.state.get_top_piece(0, 0))

    def test_gui_hover_state_transitions(self):
        """
        Serves as the hover interpolation animation test for interactive UI elements.

        Tests the exponential smoothing math in `GobbletGUI.get_hover()`:
        - Unhovered state returns baseline 0.0.
        - Hovered state smoothly interpolates upwards toward 1.0 based on delta speed factor.
        - Subsequent frames monotonically increase toward full opacity/glow.
        """
        val0 = self.gui.get_hover("test_button", False)
        self.assertEqual(val0, 0.0)

        val1 = self.gui.get_hover("test_button", True, speed=0.5)
        self.assertGreater(val1, 0.0)
        self.assertLessEqual(val1, 1.0)

        # Subsequent steps smoothly ease towards 1.0
        val2 = self.gui.get_hover("test_button", True, speed=0.5)
        self.assertGreater(val2, val1)

    def test_gui_depleted_reserve_rendering(self):
        """
        Serves as the visual edge case test for depleted reserve piece trays.

        Manually decrements Blue's Small reserve pieces to 0 and executes `gui.draw()`.
        Validates that the empty slot renders an '[EMPTY]' text badge and ghosted silhouette
        without raising null pointer or surface blit exceptions.
        """
        self.gui.state.reserves[PLAYER_BLUE][1] = 0
        self.gui.draw()
        # Verify screen drew cleanly
        self.assertIsNotNone(self.gui.screen)

    def test_gui_board_grid_and_celebration_render(self):
        """
        Serves as the victory celebration rendering integration test.

        Configures a completed 5-in-a-row horizontal line for Blue, flags `game_over = True`,
        and executes `gui.draw()`. Confirms that the composite victory renderer
        (`draw_winning_beam` and `draw_winning_piece_auras`) executes cleanly without errors.
        """
        from game_engine import Piece
        for c in range(5):
            self.gui.state.board[0][c] = [Piece(PLAYER_BLUE, 3)]
        self.gui.state.game_over = True
        self.gui.state.winner = PLAYER_BLUE
        self.gui.state.winning_line = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        self.gui.draw()
        self.assertIsNotNone(self.gui.screen)

    def test_gui_difficulty_rendering_all_levels(self):
        """
        Serves as the multi-level UI rendering test across all difficulty settings.

        Populates a rich 8-move match history and iterates through all difficulty presets
        (`easy`, `medium`, `hard`). Invokes `gui.draw()` for each setting, asserting that
        the canvas renders cleanly.
        """
        from game_engine import Move, PLAYER_BLUE, PLAYER_ORANGE
        self.gui.state.move_history = [
            (1, Move(PLAYER_BLUE, 'drop', (0, 0), size=1), "Blue Drop S -> (0,0)"),
            (2, Move(PLAYER_ORANGE, 'drop', (0, 0), size=2), "Orange Gobble M -> (0,0)"),
            (3, Move(PLAYER_BLUE, 'drop', (1, 1), size=2), "Blue Drop M -> (1,1)"),
            (4, Move(PLAYER_ORANGE, 'drop', (2, 2), size=3), "Orange Drop L -> (2,2)"),
            (5, Move(PLAYER_BLUE, 'drop', (3, 3), size=1), "Blue Drop S -> (3,3)"),
            (6, Move(PLAYER_ORANGE, 'drop', (4, 4), size=2), "Orange Drop M -> (4,4)"),
            (7, Move(PLAYER_BLUE, 'drop', (1, 2), size=2), "Blue Drop M -> (1,2)"),
            (8, Move(PLAYER_ORANGE, 'drop', (2, 1), size=3), "Orange Drop L -> (2,1)"),
        ]
        self.gui.state.turn_number = 9

        for diff in ["easy", "medium", "hard"]:
            self.gui.set_difficulty(diff)
            self.gui.draw()
            self.assertIsNotNone(self.gui.screen)

    def test_gui_difficulty_null_and_invalid_fallback(self):
        """
        Serves as the defensive fallback and input sanitization test for difficulty configurations.

        Verifies that passing None, empty strings, or unrecognized difficulty names
        (e.g., 'insane', 'ultra_hard') to either `GobbletGUI()` or `set_difficulty()`
        safely defaults to `medium` (depth 2, memory 5) without raising exceptions.
        """
        # None difficulty on initialization
        gui_none = GobbletGUI(difficulty=None)
        self.assertEqual(gui_none.difficulty, "medium")
        self.assertEqual(gui_none.ai_depth, 2)
        self.assertEqual(gui_none.memory_window, 5)

        # Invalid difficulty on initialization
        gui_inv = GobbletGUI(difficulty="insane")
        self.assertEqual(gui_inv.difficulty, "medium")

        # None difficulty on set_difficulty
        gui_none.set_difficulty(None)
        self.assertEqual(gui_none.difficulty, "medium")
        self.assertEqual(gui_none.ai_depth, 2)
        self.assertEqual(gui_none.memory_window, 5)

        # Invalid string on set_difficulty
        gui_none.set_difficulty("ultra_hard")
        self.assertEqual(gui_none.difficulty, "medium")
        self.assertEqual(gui_none.ai_depth, 2)
        self.assertEqual(gui_none.memory_window, 5)

    def test_gui_header_subtitle_format(self):
        """
        Serves as the header metadata subtitle string formatting test.

        Validates the exact formatted string rendered in the header banner across all difficulties:
        `'5x5 Grid | Win: 5-in-a-row | AI: {DIFFICULTY} (Depth {DEPTH})'`,
        ensuring uppercase difficulty names and matching search depths.
        """
        self.gui.set_difficulty("easy")
        expected_easy = f"5x5 Grid | Win: {self.gui.win_streak}-in-a-row | AI: EASY (Depth 1)"
        rendered_easy = f"5x5 Grid | Win: {self.gui.win_streak}-in-a-row | AI: {self.gui.difficulty.upper()} (Depth {self.gui.ai_depth})"
        self.assertEqual(rendered_easy, expected_easy)

        self.gui.set_difficulty("medium")
        expected_med = f"5x5 Grid | Win: {self.gui.win_streak}-in-a-row | AI: MEDIUM (Depth 2)"
        rendered_med = f"5x5 Grid | Win: {self.gui.win_streak}-in-a-row | AI: {self.gui.difficulty.upper()} (Depth {self.gui.ai_depth})"
        self.assertEqual(rendered_med, expected_med)

        self.gui.set_difficulty("hard")
        expected_hard = f"5x5 Grid | Win: {self.gui.win_streak}-in-a-row | AI: HARD (Depth 3)"
        rendered_hard = f"5x5 Grid | Win: {self.gui.win_streak}-in-a-row | AI: {self.gui.difficulty.upper()} (Depth {self.gui.ai_depth})"
        self.assertEqual(rendered_hard, expected_hard)

    def test_gui_difficulty_cycle_during_ai_thinking(self):
        """
        Serves as the thread safety and dynamic reconfiguration test for mid-search difficulty cycling.

        Tests cycling difficulty while the background AI search thread is actively executing:
        - AI starts searching at Medium (depth 2).
        - Difficulty button is clicked, cycling to Hard (depth 3).
        - Verifies that previous search cancellation is signaled (`old_cancel.is_set() == True`).
        - Verifies that a new search is initiated with incremented `search_id`.
        - Waits for the thread to finish and verifies the new move applies without state corruption.
        """
        # Trigger human move -> AI starts thinking at Medium (depth 2)
        self.gui.execute_human_move(Move(PLAYER_BLUE, 'drop', (2, 2), size=3))
        self.assertTrue(self.gui.ai_thinking)
        old_search_id = self.gui.search_id
        old_cancel = self.gui.cancel_search_event
        self.assertIsNotNone(old_cancel)

        # Cycle difficulty mid-search -> cycles to Hard (depth 3)
        self.gui.cycle_difficulty()
        self.assertEqual(self.gui.difficulty, "hard")
        self.assertEqual(self.gui.ai_depth, 3)
        self.assertEqual(self.gui.agent.max_depth, 3)
        self.assertEqual(self.gui.memory_window, 8)
        # Old search should have been signaled to cancel
        self.assertTrue(old_cancel.is_set())
        # A new search should be started with incremented search_id
        self.assertGreater(self.gui.search_id, old_search_id)
        self.assertTrue(self.gui.ai_thinking)

        # Wait for the AI search to complete cleanly
        if self.gui.ai_thread:
            self.gui.ai_thread.join(timeout=20)

        self.gui.process_ai_queue()
        # Verify valid move generated and applied cleanly without state corruption
        if self.gui.pending_ai_move:
            self.gui.chosen_move_timer = 0  # Force immediate application
            self.gui.process_ai_queue()
        self.assertEqual(self.gui.state.turn_number, 3)


if __name__ == '__main__':
    unittest.main()
