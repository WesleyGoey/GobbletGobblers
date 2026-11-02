import os
import unittest
# Force headless Pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
from game_engine import GameState, Move, PLAYER_BLUE
from gui import PygameGUI

class TestHeadlessGUI(unittest.TestCase):
    def setUp(self):
        self.state = GameState(win_streak=5)
        self.gui = PygameGUI(self.state)

    def test_gui_initialization(self):
        self.assertEqual(self.gui.cell_size, 560 // 5)
        self.assertIsNotNone(self.gui.screen)

    def test_board_render_headless(self):
        # Verify render does not crash
        self.gui.draw_board()
        self.gui.draw_sidebar_container()

if __name__ == '__main__':
    unittest.main()
