import pygame
from typing import Tuple, Optional
from game_engine import GameState, Move, PLAYER_BLUE, PLAYER_ORANGE

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

        # Selection state for mouse interaction
        self.selected_reserve_size: Optional[int] = None
        self.selected_board_pos: Optional[Tuple[int, int]] = None

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

    def draw_piece(self, surface: pygame.Surface, x: int, y: int, player: int, size: int):
        """Draw a tactile 3D Gobbler figurine with cylindrical shading and expressive eyes."""
        radius_map = {1: 18, 2: 30, 3: 44}
        height_map = {1: 28, 2: 44, 3: 60}
        radius = radius_map[size]
        height = height_map[size]

        # Colors: Blue (Human) vs Orange (AI)
        body_color = (40, 140, 255) if player == PLAYER_BLUE else (255, 120, 20)
        shadow_color = (20, 80, 160) if player == PLAYER_BLUE else (180, 70, 10)
        highlight_color = (130, 200, 255) if player == PLAYER_BLUE else (255, 180, 90)

        # Base shadow
        pygame.draw.ellipse(surface, (20, 20, 20), (x - radius, y + height//2 - 6, radius * 2, 14))

        # Cylindrical body
        rect = pygame.Rect(x - radius, y - height//2, radius * 2, height)
        pygame.draw.rect(surface, body_color, rect, border_radius=radius)
        # Highlight top rim
        pygame.draw.ellipse(surface, highlight_color, (x - radius + 3, y - height//2, radius * 2 - 6, 12))

        # Expressive cartoon eyes
        eye_y = y - height//4
        for eye_x in [x - radius//3, x + radius//3]:
            pygame.draw.circle(surface, (255, 255, 255), (eye_x, eye_y), 5)
            pygame.draw.circle(surface, (20, 20, 20), (eye_x, eye_y), 2)

    def handle_click(self, mouse_pos: Tuple[int, int]):
        """Handle player mouse clicks for selecting reserves or board relocation."""
        mx, my = mouse_pos
        # Check if clicked inside board
        if self.board_x <= mx <= self.board_x + self.board_size and self.board_y <= my <= self.board_y + self.board_size:
            col = (mx - self.board_x) // self.cell_size
            row = (my - self.board_y) // self.cell_size
            self.on_cell_clicked(row, col)

    def on_cell_clicked(self, r: int, c: int):
        if self.selected_reserve_size is not None:
            # Place selected reserve piece
            move = Move(self.state.current_player, 'drop', to_pos=(r, c), size=self.selected_reserve_size)
            if self.state.apply_move(move):
                self.selected_reserve_size = None
        else:
            # Select piece on board to move
            top = self.state.get_top_piece(r, c)
            if top and top.player == self.state.current_player:
                self.selected_board_pos = (r, c)
    
    def draw_sidebar_container(self):
        """Draw the right telemetry dashboard container."""
        sb_x = 640
        sb_y = 40
        sb_w = 600
        sb_h = 680
        # Background card
        rect = pygame.Rect(sb_x, sb_y, sb_w, sb_h)
        pygame.draw.rect(self.screen, (28, 33, 40), rect, border_radius=12)
        pygame.draw.rect(self.screen, (55, 65, 80), rect, 2, border_radius=12)

        # Header Title
        font_title = pygame.font.SysFont("Arial", 20, bold=True)
        txt = font_title.render("AI TELEMETRY & DECISION HUD", True, (240, 243, 246))
        self.screen.blit(txt, (sb_x + 20, sb_y + 18))
