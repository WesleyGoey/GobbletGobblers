import pygame
from typing import Tuple, Optional
from game_engine import GameState, PLAYER_BLUE, PLAYER_ORANGE

class PygameGUI:
    def __init__(self, state: GameState):
        pygame.init()
        self.screen_width = 1280
        self.screen_height = 760
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Gobblet Gobblers 5x5 — Team 6 AI")

        # Board layout coordinates
        self.board_x = 40
        self.board_y = 100
        self.board_size = 560
        self.cell_size = self.board_size // 5
        self.state = state

    def draw_board(self):
        """Draw the 5x5 wooden grid canvas and cell borders."""
        board_rect = pygame.Rect(self.board_x, self.board_y, self.board_size, self.board_size)
        pygame.draw.rect(self.screen, (45, 35, 25), board_rect, border_radius=12)

        for r in range(5):
            for c in range(5):
                cell_rect = pygame.Rect(
                    self.board_x + c * self.cell_size + 4,
                    self.board_y + r * self.cell_size + 4,
                    self.cell_size - 8,
                    self.cell_size - 8
                )
                pygame.draw.rect(self.screen, (60, 48, 36), cell_rect, border_radius=8)
