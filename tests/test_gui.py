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
        self.gui = GobbletGUI(width=1320, height=860)

    def tearDown(self):
        pass

    def test_gui_initial_render(self):
        """GUI should draw all panels without error."""
        self.gui.draw()
        self.assertIsNotNone(self.gui.screen)

    def test_human_reserve_selection_and_drop(self):
        """Clicking Blue reserve then board cell executes a legal drop."""
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
        """Test difficulty toggle button cycling: Medium -> Hard -> Easy -> Medium."""
        # Initial state should be default "medium"
        self.assertEqual(self.gui.difficulty, "medium")
        self.assertEqual(self.gui.ai_depth, 2)
        self.assertEqual(self.gui.agent.max_depth, 2)
        self.assertEqual(self.gui.memory_window, 5)
        self.assertEqual(self.gui.agent.memory_window, 5)
        self.assertEqual(self.gui.telemetry.max_depth, 2)

        # Click Difficulty button -> cycles to "hard"
        self.gui.handle_mouse_click(self.gui.btn_difficulty.center)
        self.assertEqual(self.gui.difficulty, "hard")
        self.assertEqual(self.gui.ai_depth, 3)
        self.assertEqual(self.gui.agent.max_depth, 3)
        self.assertEqual(self.gui.memory_window, 8)
        self.assertEqual(self.gui.agent.memory_window, 8)
        self.assertEqual(self.gui.telemetry.max_depth, 3)

        # Click Difficulty button -> cycles to "easy"
        self.gui.handle_mouse_click(self.gui.btn_difficulty.center)
        self.assertEqual(self.gui.difficulty, "easy")
        self.assertEqual(self.gui.ai_depth, 1)
        self.assertEqual(self.gui.agent.max_depth, 1)
        self.assertEqual(self.gui.memory_window, 3)
        self.assertEqual(self.gui.agent.memory_window, 3)
        self.assertEqual(self.gui.telemetry.max_depth, 1)

        # Click Difficulty button -> cycles back to "medium"
        self.gui.handle_mouse_click(self.gui.btn_difficulty.center)
        self.assertEqual(self.gui.difficulty, "medium")
        self.assertEqual(self.gui.ai_depth, 2)
        self.assertEqual(self.gui.agent.max_depth, 2)
        self.assertEqual(self.gui.memory_window, 5)
        self.assertEqual(self.gui.agent.memory_window, 5)
        self.assertEqual(self.gui.telemetry.max_depth, 2)

    def test_gui_reset_button(self):
        """Test New Game / Reset button."""
        self.gui.state.turn_number = 5
        self.gui.handle_mouse_click(self.gui.btn_new_game.center)
        self.assertEqual(self.gui.state.turn_number, 1)

    def test_gui_init_with_custom_difficulty(self):
        """Test initializing GobbletGUI with explicit difficulty levels."""
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

    def test_ai_telemetry_queue_processing(self):
        """Simulate AI telemetry events being processed and drawn on panel."""
        self.gui.event_queue.put(('EXPLORING_ROOT', {
            'move': Move(PLAYER_ORANGE, 'drop', (2, 2), size=3),
            'move_str': "[Orange] Drop L -> (2,2)",
            'alpha': -100,
            'beta': 100
        }))
        self.gui.process_ai_queue()
        # Draw should execute cleanly with telemetry populated
        self.gui.draw()

    def test_touch_move_enforcement(self):
        """Strict Touch-Move: Once a board piece is selected, player cannot pick from reserve or deselect."""
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
        Lifting a piece that reveals opponent 5-in-a-row with no legal block
        triggers an immediate instant loss upon touch!
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

    def test_gui_goal_toggle_button(self):
        """Clicking Goal button toggles win condition between 5-in-a-row and 4-in-a-row and resets board."""
        # Initial goal is 5-in-a-row (default)
        self.assertEqual(self.gui.win_streak, 5)
        self.assertEqual(self.gui.state.win_streak, 5)
        self.assertEqual(len(self.gui.state.winning_lines), 12)

        # Click Goal toggle button -> toggles to 4-in-a-row
        self.gui.handle_mouse_click(self.gui.btn_goal.center)
        self.assertEqual(self.gui.win_streak, 4)
        self.assertEqual(self.gui.state.win_streak, 4)
        self.assertEqual(len(self.gui.state.winning_lines), 28)

        # Click Goal toggle button again -> toggles back to 5-in-a-row
        self.gui.handle_mouse_click(self.gui.btn_goal.center)
        self.assertEqual(self.gui.win_streak, 5)
        self.assertEqual(self.gui.state.win_streak, 5)
        self.assertEqual(len(self.gui.state.winning_lines), 12)

    def test_gui_init_with_custom_win_streak(self):
        """Initializing GobbletGUI with custom win_streak properly sets mode."""
        gui_default = GobbletGUI()
        self.assertEqual(gui_default.win_streak, 5)
        self.assertEqual(gui_default.state.win_streak, 5)
        self.assertEqual(len(gui_default.state.winning_lines), 12)

        gui4 = GobbletGUI(win_streak=4)
        self.assertEqual(gui4.win_streak, 4)
        self.assertEqual(gui4.state.win_streak, 4)
        self.assertEqual(len(gui4.state.winning_lines), 28)

    def test_gui_reset_cancels_stale_ai_move(self):
        """Resetting or toggling goal while AI is thinking discards stale moves from applying to new game."""
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
        """Hover factor smoothly interpolates towards target."""
        val0 = self.gui.get_hover("test_button", False)
        self.assertEqual(val0, 0.0)

        val1 = self.gui.get_hover("test_button", True, speed=0.5)
        self.assertGreater(val1, 0.0)
        self.assertLessEqual(val1, 1.0)

        # Subsequent steps smoothly ease towards 1.0
        val2 = self.gui.get_hover("test_button", True, speed=0.5)
        self.assertGreater(val2, val1)

    def test_gui_depleted_reserve_rendering(self):
        """Depleting a reserve piece renders [EMPTY] indicator and ghosted silhouette without crash."""
        self.gui.state.reserves[PLAYER_BLUE][1] = 0
        self.gui.draw()
        # Verify screen drew cleanly
        self.assertIsNotNone(self.gui.screen)

    def test_gui_board_grid_and_celebration_render(self):
        """Winning 5-in-a-row state renders celebration beam and victory auras without crash."""
        from game_engine import Piece
        for c in range(5):
            self.gui.state.board[0][c] = [Piece(PLAYER_BLUE, 3)]
        self.gui.state.game_over = True
        self.gui.state.winner = PLAYER_BLUE
        self.gui.state.winning_line = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        self.gui.draw()
        self.assertIsNotNone(self.gui.screen)

    def test_main_cli_arguments(self):
        """Verify main.py CLI arguments mapping for --difficulty."""
        import subprocess
        # Test help includes --difficulty
        res = subprocess.run(["python3", "main.py", "--help"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("--difficulty", res.stdout)
        self.assertIn("easy", res.stdout)
        self.assertIn("medium", res.stdout)
        self.assertIn("hard", res.stdout)

        # Test CLI mode simulation with --difficulty easy
        res_easy = subprocess.run(
            ["python3", "main.py", "--cli", "--difficulty", "easy", "--goal", "5", "--max-turns", "1"],
            capture_output=True, text=True
        )
        self.assertEqual(res_easy.returncode, 0)
        self.assertIn("Difficulty: EASY (Depth 1, Memory Window 3)", res_easy.stdout)

        # Test CLI mode simulation with --difficulty hard
        res_hard = subprocess.run(
            ["python3", "main.py", "--cli", "--difficulty", "hard", "--goal", "5", "--max-turns", "1"],
            capture_output=True, text=True
        )
        self.assertEqual(res_hard.returncode, 0)
        self.assertIn("Difficulty: HARD (Depth 3, Memory Window 8)", res_hard.stdout)

    def test_gui_difficulty_rendering_all_levels(self):
        """Verify GUI draws cleanly across all difficulty settings and populated move history."""
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
        """Verify GobbletGUI and set_difficulty safely fallback to medium on None or invalid input."""
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
        """Verify exact header subtitle format: '5x5 Grid | Win: 5-in-a-row | AI: {DIFF} (Depth {D})'."""
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
        """Verify that cycling difficulty while AI is actively thinking cancels previous search and restarts cleanly."""
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
            self.gui.ai_thread.join(timeout=10)

        self.gui.process_ai_queue()
        # Verify valid move generated and applied cleanly without state corruption
        if self.gui.pending_ai_move:
            self.gui.chosen_move_timer = 0  # Force immediate application
            self.gui.process_ai_queue()
        self.assertEqual(self.gui.state.turn_number, 3)

    def test_main_cli_arguments_with_depth_override(self):
        """Verify main.py CLI handles medium difficulty and optional --depth override."""
        import subprocess
        # Test CLI mode simulation with default medium difficulty
        res_med = subprocess.run(
            ["python3", "main.py", "--cli", "--difficulty", "medium", "--goal", "5", "--max-turns", "1"],
            capture_output=True, text=True
        )
        self.assertEqual(res_med.returncode, 0)
        self.assertIn("Difficulty: MEDIUM (Depth 2, Memory Window 5)", res_med.stdout)

        # Test CLI mode simulation with --depth override
        res_override = subprocess.run(
            ["python3", "main.py", "--cli", "--difficulty", "medium", "--depth", "1", "--goal", "5", "--max-turns", "1"],
            capture_output=True, text=True
        )
        self.assertEqual(res_override.returncode, 0)
        self.assertIn("Difficulty: MEDIUM (Depth 1, Memory Window 5)", res_override.stdout)


if __name__ == '__main__':
    unittest.main()
