import pygame
from typing import Tuple, Optional
from game_engine import GameState, Move, PLAYER_BLUE, PLAYER_ORANGE

class PygameGUI:
    def __init__(self, state: GameState):
        from ai_agent import MinimaxAgent
        self.ai_agent = MinimaxAgent(player=PLAYER_ORANGE, max_depth=2, memory_window=5)
        self.difficulty_tier = "Medium"
        self.clock = pygame.time.Clock()

    def draw_pieces_and_reserves(self):
        """Render all top pieces on the 5x5 board, selection highlights, and bottom reserve tray."""
        font_hdr = pygame.font.SysFont("Arial", 22, bold=True)
        font_btn = pygame.font.SysFont("Arial", 14, bold=True)

        # Top Status Bar
        if self.state.game_over:
            status_txt = f"GAME OVER — PLAYER {self.state.winner} WINS! ({self.state.win_reason})"
            status_col = (255, 215, 0)
        else:
            turn_str = "HUMAN (BLUE)" if self.state.current_player == PLAYER_BLUE else "AI THINKING (ORANGE)..."
            status_txt = f"TURN {self.state.turn_number}: {turn_str}  |  Difficulty: {self.difficulty_tier}"
            status_col = (100, 200, 255) if self.state.current_player == PLAYER_BLUE else (255, 160, 60)
        self.screen.blit(font_hdr.render(status_txt, True, status_col), (40, 45))

        # Draw pieces on the 5x5 grid
        for r in range(5):
            for c in range(5):
                cx = self.board_x + c * self.cell_size + self.cell_size // 2
                cy = self.board_y + r * self.cell_size + self.cell_size // 2
                if self.selected_board_pos == (r, c):
                    sel_rect = pygame.Rect(
                        self.board_x + c * self.cell_size + 4,
                        self.board_y + r * self.cell_size + 4,
                        self.cell_size - 8,
                        self.cell_size - 8
                    )
                    pygame.draw.rect(self.screen, (255, 215, 0), sel_rect, 3, border_radius=8)
                top = self.state.get_top_piece(r, c)
                if top:
                    self.draw_piece(self.screen, cx, cy, top.player, top.size)

        # Draw Blue Reserve Tray (Bottom Left)
        tray_y = 680
        self.screen.blit(font_btn.render("YOUR RESERVES (Click to Select):", True, (200, 220, 240)), (40, tray_y - 10))
        size_labels = {1: "Small (S)", 2: "Medium (M)", 3: "Large (L)"}
        for idx, sz in enumerate([1, 2, 3]):
            rx = 290 + idx * 110
            ry = tray_y
            count = self.state.reserves[PLAYER_BLUE][sz]
            btn_rect = pygame.Rect(rx - 45, ry - 22, 95, 44)
            is_sel = (self.selected_reserve_size == sz)
            bg_col = (40, 90, 160) if is_sel else (35, 42, 55)
            pygame.draw.rect(self.screen, bg_col, btn_rect, border_radius=8)
            pygame.draw.rect(self.screen, (255, 215, 0) if is_sel else (80, 100, 130), btn_rect, 2, border_radius=8)
            lbl = f"{size_labels[sz]}: {count}"
            self.screen.blit(font_btn.render(lbl, True, (240, 245, 250)), (rx - 38, ry - 8))

        # Draw Control Buttons (Bottom Right inside Sidebar)
        controls = [("Easy", 660), ("Medium", 755), ("Hard", 850), ("Restart", 960)]
        for label, bx in controls:
            brect = pygame.Rect(bx, 665, 85, 36)
            active = (label == self.difficulty_tier)
            bcol = (200, 100, 20) if active else (45, 55, 70)
            pygame.draw.rect(self.screen, bcol, brect, border_radius=6)
            pygame.draw.rect(self.screen, (120, 140, 170), brect, 1, border_radius=6)
            self.screen.blit(font_btn.render(label, True, (255, 255, 255)), (bx + 14, 675))
        
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

    def draw_ai_decision_feed(self, ai_agent):
        """Render real-time search statistics and candidate move scores."""
        sb_x = 660
        y = 110
        font_sm = pygame.font.SysFont("Courier", 13)

        # Search Metrics
        stats = [
            f"Nodes Evaluated : {ai_agent.nodes_evaluated:,}",
            f"Branches Pruned : {ai_agent.pruned_branches:,}",
            f"Search Time     : {ai_agent.last_search_time*1000:.1f} ms",
            f"Memory Window   : {ai_agent.memory_window} turns"
        ]
        for s in stats:
            txt = font_sm.render(s, True, (160, 180, 205))
            self.screen.blit(txt, (sb_x, y))
            y += 20

        y += 10
        pygame.draw.line(self.screen, (50, 60, 75), (sb_x, y), (sb_x + 540, y), 1)
        y += 15

        # Candidate Moves Feed
        txt_hdr = font_sm.render("TOP EVALUATED CANDIDATE MOVES:", True, (255, 200, 80))
        self.screen.blit(txt_hdr, (sb_x, y))
        y += 22

        for cand in ai_agent.candidate_moves[:5]:
            m = cand['move']
            sc = cand['score']
            line = f"-> {str(m):<32} Score: {sc:>7.1f}"
            self.screen.blit(font_sm.render(line, True, (200, 220, 240)), (sb_x, y))
            y += 20

    def draw_memory_window_inspector(self, memory_records: list):
        """Draw memory status badges for buried pieces in the HUD."""
        sb_x = 660
        y = 360
        font_sm = pygame.font.SysFont("Arial", 12)
        txt_hdr = font_sm.render("AI MEMORY AUDIT (Buried Pieces):", True, (120, 220, 255))
        self.screen.blit(txt_hdr, (sb_x, y))
        y += 20

        if not memory_records:
            self.screen.blit(font_sm.render("No buried pieces on board.", True, (140, 150, 160)), (sb_x, y))
            return

        for rec in memory_records[:6]:
            r, c = rec['pos']
            rem = rec['is_remembered']
            col = (60, 200, 120) if rem else (230, 90, 80)
            tag = "REMEMBERED" if rem else "FORGOTTEN (UNKNOWN)"
            msg = f"Cell ({r},{c}) | {rec['buried_piece']} | {rec['turns_ago']}t ago -> {tag}"
            self.screen.blit(font_sm.render(msg, True, col), (sb_x, y))
            y += 18

    def restart_game(self, win_streak: int = 5):
        """Reset the match state to empty board and full 12-piece reserves."""
        self.state = GameState(win_streak=win_streak)
        self.selected_reserve_size = None
        self.selected_board_pos = None
        self.ai_agent.reset_telemetry()
        print("Game restarted successfully.")

    def draw_winning_glow(self):
        """Draw an animated golden glow line connecting the 5 winning pieces."""
        if not self.state.game_over or not self.state.winning_line:
            return

        glow_surf = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        coords = [
            (self.board_x + c * self.cell_size + self.cell_size // 2,
             self.board_y + r * self.cell_size + self.cell_size // 2)
            for r, c in self.state.winning_line
        ]
        # Draw glowing beam
        pygame.draw.lines(glow_surf, (255, 215, 0, 180), False, coords, 14)
        pygame.draw.lines(glow_surf, (255, 255, 255, 240), False, coords, 6)
        self.screen.blit(glow_surf, (0, 0))
