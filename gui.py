"""
gui.py - Pygame GUI with Interactive 5x5 Board, Reserve Trays,
and Live Real-time AI Decision Visualization Panel.

Features:
1. 5x5 Board with charming tactile Gobbler figurine visualization and tabletop fog-of-war.
2. Blue (Human) vs Orange (AI) reserve trays with piece counts and selection highlights.
3. Interactive 2-click move input with valid target destination highlighting.
4. Threaded AI execution with real-time telemetry streaming:
   - Live search tree branch path display
   - Real-time candidate move scoring
   - Alpha-Beta pruning cutoff logs
   - AI Cognitive Memory HUD (5-move working memory buffer without spoilers)
   - Final chosen move highlighted in vibrant green
5. UI controls: New Game, Goal toggle (4-in-a-row vs 5-in-a-row), AI Difficulty toggle (Easy/Medium/Hard).
"""

import pygame
import threading
import queue
import time
import math
from typing import Tuple, Optional, List, Dict, Any

from game_engine import (
    GameState, Piece, Move,
    PLAYER_BLUE, PLAYER_ORANGE, PLAYER_NAMES,
    SIZE_NAMES, SIZE_SYMBOLS
)
from ai_agent import MinimaxAgent, SearchTelemetry, MemoryInspector

# --- Color Palette (Warm Wooden & Slate Tabletop Aesthetic) ---
COLOR_BG = (18, 20, 26)             # Deep slate table surface
COLOR_PANEL_BG = (14, 16, 22)       # Quiet telemetry backdrop
COLOR_CARD_BG = (22, 26, 36)        # Refined card fill
COLOR_BORDER = (42, 48, 62)         # Crisp low-contrast border

# Board & Tabletop Wooden Tones
COLOR_BOARD_BG = (36, 28, 24)       # Warm dark mahogany board base
COLOR_BOARD_BORDER = (68, 52, 44)   # Beveled wooden outer rim
COLOR_CELL_BG = (52, 42, 36)        # Warm walnut cell tile
COLOR_CELL_LIGHT = (68, 56, 48)     # Top/left bevel highlight
COLOR_CELL_SHADOW = (34, 28, 24)    # Bottom/right bevel shadow
COLOR_CELL_HOVER = (68, 56, 48)     # Interactive hover lift
COLOR_VALID_TARGET = (34, 68, 46)   # Tactile moss green
COLOR_VALID_GLOW = (46, 204, 113)   # Radiant emerald
COLOR_SELECTED_GLOW = (245, 195, 45) # Warm amber gold
COLOR_WINNING_CELL = (90, 42, 70)   # Royal ruby winning tile

# Gobbler Player Colors
COLOR_BLUE = (38, 122, 198)         # Rich royal cobalt
COLOR_BLUE_LIGHT = (65, 165, 245)   # Volume highlight
COLOR_BLUE_DARK = (20, 65, 115)     # Deep shadow rim

COLOR_ORANGE = (235, 114, 24)       # Radiant tangerine
COLOR_ORANGE_LIGHT = (250, 150, 50) # Volume highlight
COLOR_ORANGE_DARK = (155, 62, 12)   # Deep shadow rim

# Text and Accent Colors
COLOR_TEXT_WHITE = (240, 243, 248)
COLOR_TEXT_MUTED = (135, 145, 165)
COLOR_TEXT_HIGHLIGHT = (245, 195, 45)
COLOR_GREEN_ACCENT = (46, 204, 113)
COLOR_CORAL_ACCENT = (245, 90, 75)
COLOR_RED_ACCENT = (231, 76, 60)
COLOR_PRUNED = (231, 76, 60)
COLOR_REMEMBERED = (46, 204, 113)
COLOR_FORGOTTEN = (150, 155, 165)

# Difficulty Configurations (Minimax Depth + Bounded Cognitive Memory Window)
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
DIFFICULTY_CONFIGS = {
    "easy": {
        "depth": 1,
        "memory_window": 3,
        "label": "[DIFFICULTY] EASY",
        "color": COLOR_GREEN_ACCENT,
        "bg": (20, 36, 28),
        "hover_bg": (32, 60, 46),
        "border": (40, 80, 56),
        "top_line": (30, 55, 42),
        "hover_top": (50, 95, 70),
    },
    "medium": {
        "depth": 2,
        "memory_window": 5,
        "label": "[DIFFICULTY] MEDIUM",
        "color": COLOR_TEXT_HIGHLIGHT,
        "bg": (38, 34, 22),
        "hover_bg": (60, 52, 32),
        "border": (80, 70, 44),
        "top_line": (55, 48, 30),
        "hover_top": (95, 82, 50),
    },
    "hard": {
        "depth": 3,
        "memory_window": 8,
        "label": "[DIFFICULTY] HARD",
        "color": COLOR_CORAL_ACCENT,
        "bg": (42, 24, 26),
        "hover_bg": (66, 36, 40),
        "border": (90, 48, 52),
        "top_line": (60, 32, 35),
        "hover_top": (105, 55, 60),
    },
}

PIECE_RADII = {
    1: 18,  # Small Gobbler
    2: 27,  # Medium Gobbler
    3: 37   # Large Gobbler
}


def lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    """Linearly interpolate between two RGB colors with t clamped in [0.0, 1.0]."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


class GobbletGUI:
    def __init__(self, width: int = 1320, height: int = 860, win_streak: int = 5, difficulty: str = "medium"):
        pygame.init()
        pygame.font.init()

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Gobblet Gobblers 5x5 - AI Minimax & Bounded Rationality")

        self.clock = pygame.time.Clock()
        self.running = True

        # Smooth interactive hover transition states
        self.hover_anim: Dict[str, float] = {}

        # Fonts
        self.font_title = pygame.font.SysFont("Helvetica, Arial", 24, bold=True)
        self.font_header = pygame.font.SysFont("Helvetica, Arial", 18, bold=True)
        self.font_bold = pygame.font.SysFont("Helvetica, Arial", 14, bold=True)
        self.font_btn = pygame.font.SysFont("Helvetica, Arial", 12, bold=True)
        self.font_regular = pygame.font.SysFont("Helvetica, Arial", 13)
        self.font_small = pygame.font.SysFont("Helvetica, Arial", 11)
        self.font_mono = pygame.font.SysFont("Courier, Consolas, Menlo", 12)

        # Game State
        self.win_streak: int = win_streak
        self.state = GameState(win_streak=self.win_streak)
        self.selected_reserve_size: Optional[int] = None
        self.selected_board_pos: Optional[Tuple[int, int]] = None
        self.valid_targets: List[Tuple[int, int]] = []

        # AI Agent & Difficulty Settings
        self.difficulty: str = (
            difficulty.lower() if isinstance(difficulty, str) and difficulty.lower() in DIFFICULTY_CONFIGS else "medium"
        )
        diff_cfg = DIFFICULTY_CONFIGS[self.difficulty]
        self.ai_depth: int = diff_cfg["depth"]
        self.memory_window: int = diff_cfg["memory_window"]
        self.ai_speed_mode = "Normal"  # Kept for backward compatibility
        self.speed_delays = {"Normal": 0.003, "Fast": 0.001, "Instant": 0.0}
        self.telemetry = SearchTelemetry()
        self.telemetry.max_depth = self.ai_depth
        self.agent = MinimaxAgent(
            player=PLAYER_ORANGE,
            max_depth=self.ai_depth,
            memory_window=self.memory_window,
            telemetry=self.telemetry
        )
        self.agent.step_delay = self.speed_delays[self.ai_speed_mode]

        # Threading for real-time live telemetry with search ID generation guard
        self.ai_thread: Optional[threading.Thread] = None
        self.cancel_search_event: Optional[threading.Event] = None
        self.event_queue: queue.Queue = queue.Queue()
        self.search_id: int = 0
        self.ai_thinking = False
        self.pending_ai_move: Optional[Move] = None
        self.chosen_move_timer: float = 0.0
        self.status_message: str = ""
        self.uncover_trap_warning: bool = False

        # UI Layout Bounds
        self.board_rect = pygame.Rect(140, 150, 480, 480)
        self.cell_size = 90
        self.cell_gap = 6

        # Reserve Rects (evenly spaced inside felt tray with ample breathing room)
        self.blue_reserve_rects: Dict[int, pygame.Rect] = {
            3: pygame.Rect(28, 184, 88, 136),
            2: pygame.Rect(28, 332, 88, 136),
            1: pygame.Rect(28, 480, 88, 136)
        }
        self.orange_reserve_rects: Dict[int, pygame.Rect] = {
            3: pygame.Rect(644, 184, 88, 136),
            2: pygame.Rect(644, 332, 88, 136),
            1: pygame.Rect(644, 480, 88, 136)
        }

        # Unified Control Bar Buttons: [RESET] NEW GAME, [GOAL] 5-IN-A-ROW, [DIFFICULTY] LEVEL
        self.btn_new_game = pygame.Rect(140, 646, 152, 42)
        self.btn_goal = pygame.Rect(304, 646, 152, 42)
        self.btn_win_streak = self.btn_goal  # Alias for backward compatibility
        self.btn_difficulty = pygame.Rect(468, 646, 152, 42)

        # Visualization Panel Bounds
        self.panel_rect = pygame.Rect(750, 20, 550, 820)

    def set_difficulty(self, difficulty: str):
        """Sets the AI difficulty level, updating max_depth, memory_window, and telemetry live."""
        diff = (
            difficulty.lower() if isinstance(difficulty, str) and difficulty.lower() in DIFFICULTY_CONFIGS else "medium"
        )
        self.difficulty = diff
        cfg = DIFFICULTY_CONFIGS[diff]
        self.ai_depth = cfg["depth"]
        self.memory_window = cfg["memory_window"]
        self.agent.max_depth = self.ai_depth
        self.agent.memory_window = self.memory_window
        with self.telemetry.lock:
            self.telemetry.max_depth = self.ai_depth
            self.telemetry.memory_snapshot = MemoryInspector.get_memory_snapshot(
                self.state, self.memory_window
            )

        # If AI is currently thinking, restart search immediately with new difficulty
        if self.ai_thinking and not self.state.game_over and self.state.current_player == PLAYER_ORANGE:
            self.start_ai_turn()

    def cycle_difficulty(self):
        """Cycles through EASY -> MEDIUM -> HARD -> EASY and updates agent telemetry."""
        curr_idx = DIFFICULTY_LEVELS.index(self.difficulty) if self.difficulty in DIFFICULTY_LEVELS else 1
        next_diff = DIFFICULTY_LEVELS[(curr_idx + 1) % len(DIFFICULTY_LEVELS)]
        self.set_difficulty(next_diff)

    def get_hover(self, key: str, is_hovered: bool, speed: float = 0.3) -> float:
        """Returns smoothed hover factor (0.0 to 1.0) with fast responsive ease."""
        curr = self.hover_anim.get(key, 0.0)
        target = 1.0 if is_hovered else 0.0
        val = curr + (target - curr) * speed
        if abs(val - target) < 0.01:
            val = target
        self.hover_anim[key] = val
        return val

    def run(self):
        """Main game loop running at 60 FPS."""
        while self.running:
            self.handle_events()
            self.process_ai_queue()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos
                self.handle_mouse_click(mouse_pos)

    def handle_mouse_click(self, pos: Tuple[int, int]):
        # Check Button Clicks first
        if self.btn_new_game.collidepoint(pos):
            self.reset_game()
            return

        if self.btn_goal.collidepoint(pos):
            self.win_streak = 5 if self.win_streak == 4 else 4
            self.reset_game(self.win_streak)
            return

        if self.btn_difficulty.collidepoint(pos):
            self.cycle_difficulty()
            return

        # Do not allow board interaction if game is over or AI is thinking
        if self.state.game_over or self.state.current_player != PLAYER_BLUE or self.ai_thinking:
            return

        # 1. Click on Blue Reserve Tray
        for size, rect in self.blue_reserve_rects.items():
            if rect.collidepoint(pos):
                if self.selected_board_pos is not None:
                    # Touch-move in effect: cannot pick from reserve after touching board piece!
                    self.status_message = f"Touch-Move in effect: You touched piece at {self.selected_board_pos} and must move it!"
                    return

                if self.state.reserves[PLAYER_BLUE][size] > 0:
                    if self.selected_reserve_size == size:
                        # Toggle off
                        self.selected_reserve_size = None
                        self.valid_targets = []
                        self.status_message = ""
                    else:
                        self.selected_reserve_size = size
                        self.selected_board_pos = None
                        self.calculate_valid_targets_for_drop(size)
                        self.status_message = f"Drop {SIZE_NAMES[size]} onto a highlighted green cell."
                return

        # 2. Click on Board Grid
        for r in range(5):
            for c in range(5):
                cell_rect = self.get_cell_rect(r, c)
                if cell_rect.collidepoint(pos):
                    self.handle_cell_click(r, c)
                    return

    def handle_cell_click(self, r: int, c: int):
        # Case A: A reserve piece is selected -> Try to drop
        if self.selected_reserve_size is not None:
            if (r, c) in self.valid_targets:
                move = Move(
                    player=PLAYER_BLUE,
                    move_type='drop',
                    size=self.selected_reserve_size,
                    to_pos=(r, c)
                )
                self.execute_human_move(move)
            else:
                # Cancel reserve or switch to selecting a Blue board piece
                top = self.state.get_top_piece(r, c)
                if top and top.player == PLAYER_BLUE:
                    self.selected_reserve_size = None
                    self.try_select_board_piece(r, c)
                else:
                    self.selected_reserve_size = None
                    self.valid_targets = []
                    self.status_message = ""
            return

        # Case B: A board piece is already selected (Touch-Move is active!)
        if self.selected_board_pos is not None:
            if (r, c) in self.valid_targets:
                move = Move(
                    player=PLAYER_BLUE,
                    move_type='move',
                    size=self.state.get_top_size(self.selected_board_pos[0], self.selected_board_pos[1]),
                    from_pos=self.selected_board_pos,
                    to_pos=(r, c)
                )
                self.execute_human_move(move)
            else:
                # Strict Touch-Move rule: Cannot deselect or switch pieces once touched!
                if self.uncover_trap_warning:
                    self.status_message = "[UNCOVER TRAP] You MUST place this piece to block Orange's line!"
                else:
                    self.status_message = f"Touch-Move in effect: You MUST move piece at {self.selected_board_pos} to a highlighted cell!"
            return

        # Case C: Nothing selected yet -> Touch/Select Blue piece on board
        top = self.state.get_top_piece(r, c)
        if top and top.player == PLAYER_BLUE:
            self.try_select_board_piece(r, c)

    def try_select_board_piece(self, r: int, c: int):
        """Attempts to select a board piece with Touch-Move and Uncover Trap enforcement."""
        top_piece = self.state.get_top_piece(r, c)
        if not top_piece or top_piece.player != PLAYER_BLUE:
            return

        is_trap, opp_lines = self.state.would_lift_uncover_opponent_win((r, c))
        valid_dests = self.calculate_valid_targets_for_move(r, c, is_trap)

        if is_trap:
            if not valid_dests:
                # THE UNCOVER TRAP INSTANT LOSS!
                # Player touched a piece that reveals an opponent winning line with NO legal block!
                self.state.winner = PLAYER_ORANGE
                self.state.game_over = True
                self.state.winning_line = opp_lines[0]
                self.state.win_reason = f"Uncover Trap! Blue lifted piece at ({r},{c}) revealing Orange's winning line with no block!"
                self.selected_board_pos = None
                self.valid_targets = []
                self.status_message = "[UNCOVER TRAP] Orange wins instantly!"
                return
            else:
                # Uncover trap triggered, but blocking moves exist!
                self.selected_board_pos = (r, c)
                self.valid_targets = valid_dests
                self.uncover_trap_warning = True
                self.status_message = "[UNCOVER TRAP] You MUST place this piece to block Orange's winning line!"
                return

        # Normal board piece selection
        if not valid_dests:
            self.status_message = f"Piece at ({r},{c}) has no legal moves. Select another piece."
            self.selected_board_pos = None
            self.valid_targets = []
            return

        # Touch-move engaged!
        self.selected_board_pos = (r, c)
        self.valid_targets = valid_dests
        self.uncover_trap_warning = False
        self.status_message = f"Touch-Move: Piece at ({r},{c}) locked. Click highlighted target."

    def calculate_valid_targets_for_drop(self, size: int):
        self.valid_targets = []
        for r in range(5):
            for c in range(5):
                top_size = self.state.get_top_size(r, c)
                if top_size < size:
                    self.valid_targets.append((r, c))

    def calculate_valid_targets_for_move(self, r1: int, c1: int, is_trap: bool = False) -> List[Tuple[int, int]]:
        targets = []
        top_piece = self.state.get_top_piece(r1, c1)
        if not top_piece:
            return []

        size = top_piece.size
        opponent = self.state.get_opponent(PLAYER_BLUE)

        for r2 in range(5):
            for c2 in range(5):
                if (r1, c1) == (r2, c2):
                    continue
                top_size2 = self.state.get_top_size(r2, c2)
                if top_size2 < size:
                    if is_trap:
                        # Must simulate placement to verify that all opponent lines are broken
                        p = self.state.board[r1][c1].pop()
                        self.state.board[r2][c2].append(p)
                        opp_wins = self.state.find_all_winning_lines()[opponent]
                        self.state.board[r2][c2].pop()
                        self.state.board[r1][c1].append(p)

                        if len(opp_wins) == 0:
                            targets.append((r2, c2))
                    else:
                        targets.append((r2, c2))
        return targets

    def execute_human_move(self, move: Move):
        success = self.state.apply_move(move)
        self.selected_reserve_size = None
        self.selected_board_pos = None
        self.valid_targets = []
        self.uncover_trap_warning = False
        self.status_message = ""

        if success and not self.state.game_over:
            # Trigger AI turn in background thread
            self.start_ai_turn()

    def start_ai_turn(self):
        if self.state.current_player == PLAYER_ORANGE and not self.state.game_over:
            if self.cancel_search_event:
                self.cancel_search_event.set()
            self.cancel_search_event = threading.Event()
            current_cancel = self.cancel_search_event

            self.ai_thinking = True
            self.search_id += 1
            current_search_id = self.search_id
            self.agent.step_delay = self.speed_delays[self.ai_speed_mode]

            # Clear queue
            while not self.event_queue.empty():
                try:
                    self.event_queue.get_nowait()
                except queue.Empty:
                    break

            def run_search():
                def callback(event_type, data):
                    if not current_cancel.is_set():
                        self.event_queue.put((current_search_id, event_type, data))

                best_move = self.agent.get_action(
                    self.state,
                    step_callback=callback,
                    cancel_event=current_cancel
                )
                if not current_cancel.is_set() and best_move is not None:
                    self.event_queue.put((current_search_id, 'MOVE_READY', best_move))

            self.ai_thread = threading.Thread(target=run_search, daemon=True)
            self.ai_thread.start()

    def process_ai_queue(self):
        """Processes telemetry events emitted by the AI search thread."""
        while not self.event_queue.empty():
            try:
                item = self.event_queue.get_nowait()
            except queue.Empty:
                break

            # Handle both (event_type, data) and (search_id, event_type, data)
            if len(item) == 3:
                s_id, event_type, data = item
                if s_id != self.search_id:
                    continue  # Discard stale search events from previous games
            else:
                event_type, data = item

            if event_type == 'MOVE_READY':
                if not self.state.game_over and self.state.current_player == PLAYER_ORANGE:
                    self.pending_ai_move = data
                    self.chosen_move_timer = time.time()
                self.ai_thinking = False

        # If a move is ready, show it highlighted for 0.4 seconds before applying
        if self.pending_ai_move is not None:
            if time.time() - self.chosen_move_timer > 0.4 or self.ai_speed_mode == "Instant":
                move = self.pending_ai_move
                self.pending_ai_move = None
                if move and not self.state.game_over and self.state.current_player == PLAYER_ORANGE:
                    self.state.apply_move(move)

    def reset_game(self, win_streak: Optional[int] = None):
        if self.cancel_search_event:
            self.cancel_search_event.set()
        self.search_id += 1  # Invalidate any running background search
        if win_streak is not None:
            self.win_streak = win_streak
        self.state = GameState(win_streak=self.win_streak)
        self.selected_reserve_size = None
        self.selected_board_pos = None
        self.valid_targets = []
        self.uncover_trap_warning = False
        self.status_message = ""
        self.telemetry.reset()
        self.telemetry.max_depth = self.ai_depth
        self.agent.max_depth = self.ai_depth
        self.agent.memory_window = self.memory_window
        self.ai_thinking = False
        self.pending_ai_move = None
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break

    def get_cell_rect(self, r: int, c: int) -> pygame.Rect:
        x = self.board_rect.x + c * (self.cell_size + self.cell_gap) + 4
        y = self.board_rect.y + r * (self.cell_size + self.cell_gap) + 4
        return pygame.Rect(x, y, self.cell_size, self.cell_size)

    # ==================== DRAWING ROUTINES ====================

    def draw(self):
        self.screen.fill(COLOR_BG)
        self.draw_header()
        self.draw_reserves()
        self.draw_board()
        self.draw_controls()
        self.draw_visualization_panel()

    def draw_header(self):
        # Game Title
        title_surf = self.font_title.render("GOBBLET GOBBLERS", True, COLOR_TEXT_WHITE)
        self.screen.blit(title_surf, (140, 25))

        subtitle = self.font_small.render(
            f"5x5 Grid | Win: {self.win_streak}-in-a-row | AI: {self.difficulty.upper()} (Depth {self.ai_depth})",
            True, COLOR_TEXT_MUTED
        )
        self.screen.blit(subtitle, (140, 58))

        # Turn & Status Banner
        banner_rect = pygame.Rect(140, 85, 480, 46)
        pygame.draw.rect(self.screen, COLOR_CARD_BG, banner_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_BORDER, banner_rect, 1, border_radius=8)

        # High-contrast status pill & descriptive message (>4.5:1 contrast ratio)
        if self.state.game_over:
            badge_text = "[VICTORY]"
            badge_col = COLOR_TEXT_HIGHLIGHT if self.state.winner == PLAYER_BLUE else COLOR_ORANGE_LIGHT
            msg_text = self.state.win_reason
        elif self.ai_thinking:
            badge_text = "[AI THINKING]"
            badge_col = COLOR_ORANGE_LIGHT
            msg_text = "Exploring Minimax search tree..."
        elif self.status_message:
            badge_text = "[ALERT]" if "[UNCOVER TRAP]" in self.status_message else "[TOUCH-MOVE]"
            badge_col = COLOR_RED_ACCENT if "[UNCOVER TRAP]" in self.status_message else COLOR_BLUE_LIGHT
            msg_text = self.status_message.strip()
        elif self.state.current_player == PLAYER_BLUE:
            badge_text = "[HUMAN TURN]"
            badge_col = COLOR_BLUE_LIGHT
            if self.selected_reserve_size:
                msg_text = f"Drop {SIZE_NAMES[self.selected_reserve_size]} on highlighted green cell"
            elif self.selected_board_pos:
                msg_text = f"Move piece from {self.selected_board_pos} to target cell"
            else:
                msg_text = "Click a reserve tray piece or a board piece"
        else:
            badge_text = "[AI TURN]"
            badge_col = COLOR_ORANGE_LIGHT
            msg_text = "Awaiting decision..."

        # Status badge pill
        b_surf = self.font_bold.render(badge_text, True, badge_col)
        pill_rect = pygame.Rect(banner_rect.x + 10, banner_rect.y + 10, b_surf.get_width() + 14, 26)
        pygame.draw.rect(self.screen, (28, 34, 46), pill_rect, border_radius=6)
        pygame.draw.rect(self.screen, badge_col, pill_rect, 1, border_radius=6)
        self.screen.blit(b_surf, (pill_rect.centerx - b_surf.get_width() // 2, pill_rect.centery - b_surf.get_height() // 2))

        # Status message with high contrast against card background (>10:1 ratio)
        m_surf = self.font_regular.render(msg_text, True, COLOR_TEXT_WHITE)
        self.screen.blit(m_surf, (pill_rect.right + 12, banner_rect.centery - m_surf.get_height() // 2))

    def draw_reserves(self):
        mouse_pos = pygame.mouse.get_pos()
        pulse = 0.5 + 0.5 * math.sin(time.time() * 6.0)

        # 1. Blue Reserve Tray (Left) - Navy felt-lined tray
        tray_blue = pygame.Rect(20, 136, 105, 494)
        pygame.draw.rect(self.screen, (20, 26, 36), tray_blue, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_BLUE_DARK, tray_blue, 2, border_radius=10)

        lbl_blue = self.font_bold.render("BLUE", True, COLOR_BLUE_LIGHT)
        self.screen.blit(lbl_blue, (tray_blue.centerx - lbl_blue.get_width() // 2, tray_blue.y + 8))
        lbl_sub = self.font_small.render("(Human)", True, COLOR_TEXT_MUTED)
        self.screen.blit(lbl_sub, (tray_blue.centerx - lbl_sub.get_width() // 2, tray_blue.y + 24))

        size_labels = {3: "LARGE", 2: "MEDIUM", 1: "SMALL"}
        can_interact = (not self.ai_thinking and not self.state.game_over and self.state.current_player == PLAYER_BLUE and self.selected_board_pos is None)

        for size, rect in self.blue_reserve_rects.items():
            count = self.state.reserves[PLAYER_BLUE][size]
            is_selected = (self.selected_reserve_size == size)
            is_hovered = rect.collidepoint(mouse_pos) and (count > 0) and can_interact
            hov_val = self.get_hover(f"blue_card_{size}", is_hovered)

            # Card background with tactile bevel and smooth hover transition
            if is_selected:
                bg_col = (36, 52, 78)
                border_col = COLOR_SELECTED_GLOW
                border_w = 3
            elif count == 0:
                bg_col = (20, 24, 32)
                border_col = (34, 40, 52)
                border_w = 1
            else:
                bg_col = lerp_color((28, 34, 46), (42, 56, 78), hov_val)
                border_col = lerp_color(COLOR_BORDER, (80, 140, 220), hov_val)
                border_w = 2 if hov_val > 0.4 else 1

            pygame.draw.rect(self.screen, bg_col, rect, border_radius=8)
            pygame.draw.rect(self.screen, border_col, rect, border_w, border_radius=8)

            # Top highlight line on card
            hi_line = lerp_color((40, 48, 64), (65, 85, 118), hov_val if count > 0 else 0.0)
            pygame.draw.line(self.screen, hi_line, (rect.left + 4, rect.top + 2), (rect.right - 4, rect.top + 2), 1)

            # Figurine size label at top (ample breathing room)
            lbl_col = COLOR_BLUE_LIGHT if (is_selected or hov_val > 0.4) else (COLOR_TEXT_MUTED if count > 0 else (90, 100, 115))
            lbl_s = self.font_small.render(size_labels[size], True, lbl_col)
            self.screen.blit(lbl_s, (rect.centerx - lbl_s.get_width() // 2, rect.y + 10))

            center = (rect.centerx, rect.y + 70)

            # Glowing amber selection halo around active figurine
            if is_selected:
                sel_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                halo_r = PIECE_RADII[size] + 8 + int(3 * pulse)
                pygame.draw.circle(sel_surf, (245, 195, 45, int(80 + 50 * pulse)), center, halo_r)
                self.screen.blit(sel_surf, (0, 0))

            # Draw tactile Gobbler figurine (dimmed ghosted if depleted)
            self.draw_piece_3d(center, size, PLAYER_BLUE, is_top=True, is_disabled=(count == 0))

            # Label & Count
            if count == 0:
                cnt_text = "[EMPTY]"
                cnt_col = (110, 120, 135)
            else:
                cnt_text = f"x {count}"
                cnt_col = COLOR_TEXT_WHITE if not is_hovered else (255, 255, 255)
            txt_surf = self.font_bold.render(cnt_text, True, cnt_col)
            self.screen.blit(txt_surf, (rect.centerx - txt_surf.get_width() // 2, rect.bottom - 20))

        # 2. Orange Reserve Tray (Right of board) - Terracotta felt-lined tray
        tray_orange = pygame.Rect(635, 136, 105, 494)
        pygame.draw.rect(self.screen, (34, 25, 20), tray_orange, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_ORANGE_DARK, tray_orange, 2, border_radius=10)

        lbl_orange = self.font_bold.render("ORANGE", True, COLOR_ORANGE_LIGHT)
        self.screen.blit(lbl_orange, (tray_orange.centerx - lbl_orange.get_width() // 2, tray_orange.y + 8))
        lbl_ai = self.font_small.render("(AI Agent)", True, COLOR_TEXT_MUTED)
        self.screen.blit(lbl_ai, (tray_orange.centerx - lbl_ai.get_width() // 2, tray_orange.y + 24))

        for size, rect in self.orange_reserve_rects.items():
            count = self.state.reserves[PLAYER_ORANGE][size]
            bg_col = (36, 28, 24) if count > 0 else (26, 22, 20)
            b_col = COLOR_BORDER if count > 0 else (45, 36, 32)
            pygame.draw.rect(self.screen, bg_col, rect, border_radius=8)
            pygame.draw.rect(self.screen, b_col, rect, 1, border_radius=8)

            lbl_col = COLOR_TEXT_MUTED if count > 0 else (95, 85, 80)
            lbl_s = self.font_small.render(size_labels[size], True, lbl_col)
            self.screen.blit(lbl_s, (rect.centerx - lbl_s.get_width() // 2, rect.y + 10))

            center = (rect.centerx, rect.y + 70)
            self.draw_piece_3d(center, size, PLAYER_ORANGE, is_top=True, is_disabled=(count == 0))

            if count == 0:
                cnt_text = "[EMPTY]"
                cnt_col = (110, 95, 90)
            else:
                cnt_text = f"x {count}"
                cnt_col = COLOR_TEXT_WHITE
            txt_surf = self.font_bold.render(cnt_text, True, cnt_col)
            self.screen.blit(txt_surf, (rect.centerx - txt_surf.get_width() // 2, rect.bottom - 20))

    def draw_board(self):
        # 1. Outer Board frame - warm dark mahogany with bevel and subtle grain lines
        pygame.draw.rect(self.screen, COLOR_BOARD_BG, self.board_rect, border_radius=14)
        pygame.draw.rect(self.screen, COLOR_BOARD_BORDER, self.board_rect, 3, border_radius=14)

        # Subtle horizontal wood grain accent lines across border margin
        for gy in range(self.board_rect.top + 8, self.board_rect.bottom - 8, 16):
            pygame.draw.line(self.screen, (44, 34, 28), (self.board_rect.left + 6, gy), (self.board_rect.right - 6, gy), 1)

        # 2. Subtle Carved Grid Inlay Lines (between cell columns and rows)
        for i in range(1, 5):
            # Vertical groove between column i-1 and column i
            c_left = self.get_cell_rect(0, i - 1).right
            c_right = self.get_cell_rect(0, i).left
            gx = (c_left + c_right) // 2
            pygame.draw.line(self.screen, (22, 16, 12), (gx, self.board_rect.top + 8), (gx, self.board_rect.bottom - 8), 2)
            pygame.draw.line(self.screen, (52, 40, 32), (gx + 1, self.board_rect.top + 8), (gx + 1, self.board_rect.bottom - 8), 1)

            # Horizontal groove between row i-1 and row i
            r_top = self.get_cell_rect(i - 1, 0).bottom
            r_bottom = self.get_cell_rect(i, 0).top
            gy = (r_top + r_bottom) // 2
            pygame.draw.line(self.screen, (22, 16, 12), (self.board_rect.left + 8, gy), (self.board_rect.right - 8, gy), 2)
            pygame.draw.line(self.screen, (52, 40, 32), (self.board_rect.left + 8, gy + 1), (self.board_rect.right - 8, gy + 1), 1)

        mouse_pos = pygame.mouse.get_pos()
        pulse = 0.5 + 0.5 * math.sin(time.time() * 6.5)

        # 3. Draw Cell Tile Floors & Recessed Bevels
        for r in range(5):
            for c in range(5):
                cell_rect = self.get_cell_rect(r, c)
                is_hovered = cell_rect.collidepoint(mouse_pos)
                hov_val = self.get_hover(f"cell_{r}_{c}", is_hovered)
                is_valid = (r, c) in self.valid_targets
                is_selected = (self.selected_board_pos == (r, c))
                is_winning_cell = bool(self.state.winning_line and (r, c) in self.state.winning_line)

                # Cell background - warm wooden tile palette with recessed bevel
                if is_winning_cell:
                    cell_col = (90, 42, 70)
                    bevel_shadow = (60, 25, 45)
                    bevel_light = (140, 70, 115)
                elif is_valid:
                    # Hovering over a valid drop target illuminates it with emerald radiance
                    cell_col = lerp_color((34, 68, 46), (48, 96, 64), hov_val)
                    bevel_shadow = (22, 48, 32)
                    bevel_light = (60, 120, 80)
                elif is_selected:
                    cell_col = (72, 60, 28)
                    bevel_shadow = (48, 40, 18)
                    bevel_light = (110, 95, 45)
                else:
                    cell_col = lerp_color((52, 42, 36), (72, 60, 52), hov_val)
                    bevel_shadow = (34, 28, 24)
                    bevel_light = lerp_color((66, 54, 46), (88, 74, 64), hov_val)

                pygame.draw.rect(self.screen, cell_col, cell_rect, border_radius=8)

                # Tactile recessed 3D tile bevel (light from top-left: top/left cast shadow, bottom/right illuminated)
                pygame.draw.line(self.screen, bevel_shadow, (cell_rect.left + 2, cell_rect.top + 1), (cell_rect.right - 2, cell_rect.top + 1), 2)
                pygame.draw.line(self.screen, bevel_shadow, (cell_rect.left + 1, cell_rect.top + 2), (cell_rect.left + 1, cell_rect.bottom - 2), 2)
                pygame.draw.line(self.screen, bevel_light, (cell_rect.left + 2, cell_rect.bottom - 1), (cell_rect.right - 2, cell_rect.bottom - 1), 2)
                pygame.draw.line(self.screen, bevel_light, (cell_rect.right - 1, cell_rect.top + 2), (cell_rect.right - 1, cell_rect.bottom - 2), 2)

                # Cell Border / Highlight
                if is_winning_cell:
                    pygame.draw.rect(self.screen, (255, 215, 60), cell_rect, 3, border_radius=8)
                elif is_selected:
                    pygame.draw.rect(self.screen, COLOR_SELECTED_GLOW, cell_rect, 3, border_radius=8)
                elif is_valid:
                    valid_border_col = (
                        int(35 + 25 * pulse),
                        int(180 + 45 * pulse),
                        int(90 + 35 * pulse)
                    )
                    border_w = 3 if hov_val > 0.5 else 2
                    pygame.draw.rect(self.screen, valid_border_col, cell_rect, border_w, border_radius=8)
                else:
                    b_col = lerp_color((42, 34, 28), (80, 66, 56), hov_val)
                    pygame.draw.rect(self.screen, b_col, cell_rect, 1, border_radius=8)

                # High-clarity coordinate markers (drawn on tile floor BEFORE pieces)
                coord_col = (150, 136, 124) if hov_val > 0.5 else (125, 112, 100)
                coord_surf = self.font_small.render(f"{r},{c}", True, coord_col)
                self.screen.blit(coord_surf, (cell_rect.x + 6, cell_rect.y + 5))

        # 4. Celebratory radiant winning beam & golden halos ON BOARD TILES (underneath pieces)
        if self.state.winning_line:
            self.draw_winning_beam(self.state.winning_line)

        # 5. Draw Pieces inside cells
        for r in range(5):
            for c in range(5):
                cell_rect = self.get_cell_rect(r, c)
                is_selected = (self.selected_board_pos == (r, c))
                stack = self.state.get_cell_stack(r, c)

                # If selected piece on board: draw pulsing golden aura behind piece
                if is_selected and stack:
                    sel_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                    halo_r = PIECE_RADII.get(stack[-1].size, 27) + 8 + int(3 * pulse)
                    pygame.draw.circle(sel_surf, (245, 195, 45, int(75 + 45 * pulse)), cell_rect.center, halo_r)
                    self.screen.blit(sel_surf, (0, 0))

                if stack:
                    self.draw_cell_stack(cell_rect.center, stack)

        # 6. Celebratory golden victory rings and sparkling victory stars around/above winning figurines
        if self.state.winning_line:
            self.draw_winning_piece_auras(self.state.winning_line)

        # 7. Valid target indicators (pulsing emerald drop indicators)
        for r in range(5):
            for c in range(5):
                cell_rect = self.get_cell_rect(r, c)
                is_valid = (r, c) in self.valid_targets
                if not is_valid:
                    continue
                is_hovered = cell_rect.collidepoint(mouse_pos)
                stack = self.state.get_cell_stack(r, c)

                if stack:
                    # GOBBLE TARGET: occupied by smaller piece!
                    gobble_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                    ring_r = PIECE_RADII.get(stack[-1].size, 25) + 6 + int(3 * pulse)
                    reticle_col = (46, 204, 113, int(150 + 90 * pulse)) if not is_hovered else (255, 215, 60, int(190 + 60 * pulse))
                    pygame.draw.circle(gobble_surf, reticle_col, cell_rect.center, ring_r, 2 if not is_hovered else 3)
                    self.screen.blit(gobble_surf, (0, 0))
                    g_col = COLOR_VALID_GLOW if not is_hovered else COLOR_TEXT_HIGHLIGHT
                    g_lbl = self.font_small.render("[GOBBLE]", True, g_col)
                    self.screen.blit(g_lbl, (cell_rect.centerx - g_lbl.get_width() // 2, cell_rect.bottom - 16))
                else:
                    # Empty cell landing marker:
                    glow_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                    outer_r = (14 if is_hovered else 11) + int(3 * pulse)
                    pygame.draw.circle(glow_surf, (46, 204, 113, int(70 + 60 * pulse)), cell_rect.center, outer_r)
                    self.screen.blit(glow_surf, (0, 0))
                    core_r = 8 if is_hovered else 6
                    pygame.draw.circle(self.screen, COLOR_VALID_GLOW, cell_rect.center, core_r)
                    pygame.draw.circle(self.screen, (220, 255, 240), cell_rect.center, 3)

    def draw_winning_beam(self, winning_line: List[Tuple[int, int]]):
        """
        Renders radiant golden celebratory connection beam linking all winning cells
        directly on the board tile surface underneath the pieces.
        """
        if not winning_line or len(winning_line) < 2:
            return

        line_sorted = sorted(winning_line, key=lambda rc: (rc[0], rc[1]))
        centers = [self.get_cell_rect(r, c).center for r, c in line_sorted]
        pulse = 0.5 + 0.5 * math.sin(time.time() * 5.5)

        glow_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # 1. Radiant beam on tile floor
        for i in range(len(centers) - 1):
            p1 = centers[i]
            p2 = centers[i + 1]
            pygame.draw.line(glow_surf, (255, 215, 50, int(70 + 50 * pulse)), p1, p2, 20)
            pygame.draw.line(glow_surf, (255, 230, 90, int(160 + 70 * pulse)), p1, p2, 9)
            pygame.draw.line(glow_surf, (255, 255, 240, 245), p1, p2, 3)

        # 2. Golden halo on each winning tile floor
        for r, c in winning_line:
            cx, cy = self.get_cell_rect(r, c).center
            halo_r = 44 + int(4 * pulse)
            pygame.draw.circle(glow_surf, (255, 215, 0, int(50 + 35 * pulse)), (cx, cy), halo_r)
            pygame.draw.circle(glow_surf, (255, 235, 120, int(80 + 45 * pulse)), (cx, cy), halo_r - 8)

        self.screen.blit(glow_surf, (0, 0))

    def draw_winning_piece_auras(self, winning_line: List[Tuple[int, int]]):
        """
        Renders celebratory golden victory aura rings around the perimeter of each
        winning Gobbler piece and sparkling diamond stars above their crown tufts.
        Keeps Gobbler facial features and glossy 3D volumes 100% visible and vibrant!
        """
        if not winning_line:
            return

        pulse = 0.5 + 0.5 * math.sin(time.time() * 5.5)
        glow_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        for r, c in winning_line:
            cell_rect = self.get_cell_rect(r, c)
            cx, cy = cell_rect.center
            top_piece = self.state.get_top_piece(r, c)
            piece_r = PIECE_RADII.get(top_piece.size, 27) if top_piece else 27

            # Golden victory aura ring around figurine perimeter
            ring_r = piece_r + 6 + int(3 * pulse)
            pygame.draw.circle(glow_surf, (255, 215, 45, int(140 + 80 * pulse)), (cx, cy), ring_r, 2)
            pygame.draw.circle(glow_surf, (255, 235, 120, int(60 + 40 * pulse)), (cx, cy), ring_r + 3, 1)

            # Sparkling diamond star above crown tufts
            star_y = cy - piece_r - (8 if (top_piece and top_piece.size == 3) else 6)
            star_len = 5 + int(2 * pulse)
            pygame.draw.polygon(glow_surf, (255, 255, 240, 240), [
                (cx, star_y - star_len),
                (cx + star_len // 2, star_y),
                (cx, star_y + star_len),
                (cx - star_len // 2, star_y)
            ])
            pygame.draw.circle(glow_surf, (255, 255, 255, 250), (cx, star_y), 2)

        self.screen.blit(glow_surf, (0, 0))

    def draw_winning_celebration(self, winning_line: List[Tuple[int, int]]):
        """Maintained for backwards compatibility."""
        self.draw_winning_beam(winning_line)
        self.draw_winning_piece_auras(winning_line)

    def draw_cell_stack(self, center: Tuple[int, int], stack: List[Piece]):
        """
        Draws the top piece on the cell.
        Tabletop fidelity: Hidden pieces underneath are completely concealed (no elevation rings, no depth badges).
        """
        if not stack:
            return
        top_piece = stack[-1]
        self.draw_piece_3d(center, top_piece.size, top_piece.player, is_top=True)

    def draw_piece_3d(self, center: Tuple[int, int], size: int, player: int, is_top: bool = True, is_disabled: bool = False):
        """
        Draws a charming, tactile Gobbler figurine inspired by the tabletop board game.
        Features:
        - Soft translucent drop shadow on board/tray surface
        - Cute sculpted top tufts with top-left specular highlights
        - Spherical dome volume gradient (base rim, body fill, volume highlights, specular glint)
        - Subtle warm rosy cheeks
        - Whimsical cartoon Gobbler eyes (white sclera, dark pupils, lively specular catchlights)
        - Characteristic expressions scaled per piece size (open gobble mouth, friendly grin, cute baby 'o')
        - Disabled / ghosted silhouette support for depleted reserve slots
        """
        cx, cy = center
        radius = PIECE_RADII.get(size, 27)

        # 1. Soft Translucent Drop Shadow (creates physical depth & separation)
        shadow_surf = pygame.Surface((radius * 2 + 18, radius * 2 + 18), pygame.SRCALPHA)
        s_alpha1 = 25 if is_disabled else 65
        s_alpha2 = 45 if is_disabled else 105
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, s_alpha1), (4, 10, radius * 2 + 8, radius * 2 - 4))
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, s_alpha2), (6, 12, radius * 2 + 4, radius * 2 - 8))
        self.screen.blit(shadow_surf, (cx - radius - 8, cy - radius - 6))

        # 2. Spherical Dome Volume Gradient & Tufts Palettes
        if is_disabled:
            # Ghosted / depleted palette for empty reserve slot
            base_rim = (34, 40, 50)
            body_col = (42, 48, 60)
            vol_col = (52, 60, 74)
            hi_col = (65, 75, 92)
            shine_col = (90, 105, 125)
            blush_col = (0, 0, 0, 0)
        elif player == PLAYER_BLUE:
            base_rim = COLOR_BLUE_DARK
            body_col = COLOR_BLUE
            vol_col = (55, 145, 230)
            hi_col = (90, 175, 250)
            shine_col = (235, 248, 255)
            blush_col = (50, 160, 240, 55)
        else:
            base_rim = COLOR_ORANGE_DARK
            body_col = COLOR_ORANGE
            vol_col = (248, 135, 42)
            hi_col = (255, 180, 92)
            shine_col = (255, 248, 230)
            blush_col = (255, 120, 60, 55)

        # 3. Cute Sculpted Top Tufts (Distinctive tactile Gobbler silhouette with directional glints!)
        tuft_r = 3 if size == 1 else (4 if size == 2 else 5)
        tuft_y = cy - radius + 1
        if size == 1:
            pygame.draw.circle(self.screen, base_rim, (cx, tuft_y - tuft_r), tuft_r + 1)
            pygame.draw.circle(self.screen, vol_col, (cx, tuft_y - tuft_r), tuft_r)
            if not is_disabled:
                pygame.draw.circle(self.screen, shine_col, (cx - 1, tuft_y - tuft_r - 1), 1)
        elif size == 2:
            for tox in [-4, 4]:
                pygame.draw.circle(self.screen, base_rim, (cx + tox, tuft_y - tuft_r), tuft_r + 1)
                pygame.draw.circle(self.screen, vol_col, (cx + tox, tuft_y - tuft_r), tuft_r)
                if not is_disabled:
                    pygame.draw.circle(self.screen, shine_col, (cx + tox - 1, tuft_y - tuft_r - 1), 1)
        elif size == 3:
            for tox, toy in [(-8, 1), (0, -2), (8, 1)]:
                pygame.draw.circle(self.screen, base_rim, (cx + tox, tuft_y - tuft_r + toy), tuft_r + 1)
                pygame.draw.circle(self.screen, vol_col, (cx + tox, tuft_y - tuft_r + toy), tuft_r)
                if not is_disabled:
                    pygame.draw.circle(self.screen, shine_col, (cx + tox - 1, tuft_y - tuft_r + toy - 1), 1)

        # 4. Main Dome Body & Volume
        pygame.draw.circle(self.screen, base_rim, center, radius)
        pygame.draw.circle(self.screen, body_col, center, radius - 1)

        ox = cx - int(radius * 0.20)
        oy = cy - int(radius * 0.20)
        pygame.draw.circle(self.screen, vol_col, (ox, oy), int(radius * 0.82))
        pygame.draw.circle(self.screen, hi_col, (ox - 1, oy - 1), int(radius * 0.54))

        # Specular glint
        glint_x = cx - int(radius * 0.42)
        glint_y = cy - int(radius * 0.42)
        pygame.draw.circle(self.screen, shine_col, (glint_x, glint_y), max(2, int(radius * 0.18)))

        if is_disabled:
            return  # Ghosted silhouette has no eyes or mouth

        # 5. Cute Soft Rosy Cheeks
        cheek_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        blush_r = 3 if size == 1 else (5 if size == 2 else 7)
        cheek_dx = int(radius * 0.55)
        cheek_dy = int(radius * 0.22)
        pygame.draw.circle(cheek_surf, blush_col, (radius - cheek_dx, radius + cheek_dy), blush_r)
        pygame.draw.circle(cheek_surf, blush_col, (radius + cheek_dx, radius + cheek_dy), blush_r)
        self.screen.blit(cheek_surf, (cx - radius, cy - radius))

        # 6. Whimsical Cartoon Gobbler Eyes
        if size == 3:  # Large
            eye_dx, eye_dy = 11, -8
            sclera_r, pupil_r, cl_r = 8.0, 4.0, 1.6
        elif size == 2:  # Medium
            eye_dx, eye_dy = 8, -6
            sclera_r, pupil_r, cl_r = 6.0, 3.0, 1.3
        else:  # Small
            eye_dx, eye_dy = 5, -4
            sclera_r, pupil_r, cl_r = 4.2, 2.0, 1.0

        for is_left, ex in [(True, cx - eye_dx), (False, cx + eye_dx)]:
            ey = cy + eye_dy
            pygame.draw.circle(self.screen, (20, 24, 30), (int(ex), int(ey)), int(sclera_r + 1.2))
            pygame.draw.circle(self.screen, (254, 254, 255), (int(ex), int(ey)), int(sclera_r))
            pox = ex + (0.6 if is_left else -0.6)
            poy = ey + 0.6
            pygame.draw.circle(self.screen, (20, 24, 32), (int(pox), int(poy)), int(pupil_r))
            cl_x = pox - pupil_r * 0.35
            cl_y = poy - pupil_r * 0.35
            pygame.draw.circle(self.screen, (255, 255, 255), (int(cl_x), int(cl_y)), max(1, int(cl_r)))

        # 7. Expressive Character Detailing
        if size == 3:
            # Large Gobbler: Wide happy open gobble mouth with dark interior and pink tongue!
            mouth_surf = pygame.Surface((24, 16), pygame.SRCALPHA)
            pygame.draw.ellipse(mouth_surf, (30, 20, 22), (2, 2, 20, 12))
            pygame.draw.ellipse(mouth_surf, (230, 100, 120), (6, 7, 12, 7))
            pygame.draw.arc(mouth_surf, (15, 12, 14), (2, 2, 20, 12), math.pi * 0.1, math.pi * 0.9, 2)
            self.screen.blit(mouth_surf, (cx - 12, cy + 3))
        elif size == 2:
            # Medium Gobbler: Friendly curious smile
            m_rect = pygame.Rect(cx - 7, cy + 3, 14, 10)
            pygame.draw.arc(self.screen, (24, 20, 20), m_rect, math.pi * 1.08, math.pi * 1.92, 2)
        else:
            # Small Gobbler: Cute little open 'o' mouth
            pygame.draw.circle(self.screen, (24, 20, 20), (cx, cy + 5), 2.5)
            pygame.draw.circle(self.screen, (240, 110, 130), (cx, cy + 5), 1.5)

    def draw_controls(self):
        mouse_pos = pygame.mouse.get_pos()

        # 1. New Game Button
        hov1 = self.btn_new_game.collidepoint(mouse_pos)
        hov1_val = self.get_hover("btn_new_game", hov1)
        bg1 = lerp_color((30, 36, 48), (50, 62, 82), hov1_val)
        border1 = lerp_color(COLOR_BORDER, (110, 150, 210), hov1_val)
        pygame.draw.rect(self.screen, bg1, self.btn_new_game, border_radius=8)
        pygame.draw.rect(self.screen, border1, self.btn_new_game, 1, border_radius=8)
        top_line1 = lerp_color((42, 50, 66), (75, 92, 120), hov1_val)
        pygame.draw.line(self.screen, top_line1,
                         (self.btn_new_game.left + 4, self.btn_new_game.top + 2),
                         (self.btn_new_game.right - 4, self.btn_new_game.top + 2), 1)
        lbl1 = self.font_btn.render("[RESET] NEW GAME", True, COLOR_TEXT_WHITE)
        self.screen.blit(lbl1, (self.btn_new_game.centerx - lbl1.get_width() // 2, self.btn_new_game.centery - lbl1.get_height() // 2))

        # 2. Goal Toggle Button (5-in-a-row vs 4-in-a-row)
        hov_goal = self.btn_goal.collidepoint(mouse_pos)
        hov_goal_val = self.get_hover("btn_goal", hov_goal)
        bg_goal = lerp_color((36, 32, 26), (60, 54, 40), hov_goal_val)
        border_goal = lerp_color((80, 70, 48), COLOR_SELECTED_GLOW, hov_goal_val)
        pygame.draw.rect(self.screen, bg_goal, self.btn_goal, border_radius=8)
        pygame.draw.rect(self.screen, border_goal, self.btn_goal, 1, border_radius=8)
        top_line_goal = lerp_color((55, 48, 34), (110, 96, 60), hov_goal_val)
        pygame.draw.line(self.screen, top_line_goal,
                         (self.btn_goal.left + 4, self.btn_goal.top + 2),
                         (self.btn_goal.right - 4, self.btn_goal.top + 2), 1)
        lbl_goal_text = f"[GOAL] {self.win_streak}-IN-A-ROW"
        lbl_goal = self.font_btn.render(lbl_goal_text, True, COLOR_TEXT_HIGHLIGHT)
        self.screen.blit(lbl_goal, (self.btn_goal.centerx - lbl_goal.get_width() // 2, self.btn_goal.centery - lbl_goal.get_height() // 2))

        # 3. Difficulty Button (EASY / MEDIUM / HARD)
        hov_diff = self.btn_difficulty.collidepoint(mouse_pos)
        hov_diff_val = self.get_hover("btn_difficulty", hov_diff)
        diff_cfg = DIFFICULTY_CONFIGS[self.difficulty]
        diff_accent = diff_cfg["color"]
        bg_diff = lerp_color(diff_cfg["bg"], diff_cfg["hover_bg"], hov_diff_val)
        border_diff = lerp_color(diff_cfg["border"], diff_accent, hov_diff_val)
        top_line_diff = lerp_color(diff_cfg["top_line"], diff_cfg["hover_top"], hov_diff_val)

        pygame.draw.rect(self.screen, bg_diff, self.btn_difficulty, border_radius=8)
        pygame.draw.rect(self.screen, border_diff, self.btn_difficulty, 1, border_radius=8)
        pygame.draw.line(self.screen, top_line_diff,
                         (self.btn_difficulty.left + 4, self.btn_difficulty.top + 2),
                         (self.btn_difficulty.right - 4, self.btn_difficulty.top + 2), 1)
        lbl_diff = self.font_btn.render(diff_cfg["label"], True, diff_accent)
        self.screen.blit(lbl_diff, (self.btn_difficulty.centerx - lbl_diff.get_width() // 2,
                                    self.btn_difficulty.centery - lbl_diff.get_height() // 2))

    # ==================== LIVE AI VISUALIZATION PANEL ====================

    def draw_visualization_panel(self):
        """
        Renders the rich real-time AI decision-making telemetry panel.
        Contains:
        1. AI Search & Bounded Memory Status Banner
        2. Real-time Search Tree & Branch Trail
        3. Candidate Moves with evaluated scores & [PRUNED] tags
        4. Memory Belief Table (What AI Remembers vs Forgot)
        """
        # Outer panel frame
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, self.panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_BORDER, self.panel_rect, 2, border_radius=12)

        x = self.panel_rect.x + 16
        y = self.panel_rect.y + 16

        # --- SECTION 1: HEADER & TELEMETRY ---
        title = self.font_header.render("AI DECISION TELEMETRY (LIVE)", True, COLOR_TEXT_HIGHLIGHT)
        self.screen.blit(title, (x, y))

        status_text = "THINKING..." if self.ai_thinking else "IDLE (Awaiting Human)"
        status_col = COLOR_ORANGE_LIGHT if self.ai_thinking else COLOR_GREEN_ACCENT
        status_surf = self.font_bold.render(status_text, True, status_col)
        pill_w = status_surf.get_width() + 26
        pill_rect = pygame.Rect(self.panel_rect.right - pill_w - 20, y, pill_w, 24)
        pygame.draw.rect(self.screen, (22, 28, 38), pill_rect, border_radius=12)
        pygame.draw.rect(self.screen, status_col, pill_rect, 1, border_radius=12)
        pygame.draw.circle(self.screen, status_col, (pill_rect.x + 10, pill_rect.centery), 4)
        self.screen.blit(status_surf, (pill_rect.x + 18, pill_rect.centery - status_surf.get_height() // 2))
        y += 32

        # Telemetry metrics bar
        metrics_rect = pygame.Rect(x, y, self.panel_rect.width - 32, 54)
        pygame.draw.rect(self.screen, COLOR_CARD_BG, metrics_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_BORDER, metrics_rect, 1, border_radius=6)

        m_line1 = f"Depth: {self.telemetry.max_depth} plies  |  Nodes Visited: {self.telemetry.nodes_visited}  |  Pruned: {self.telemetry.prune_count}"
        t1 = self.font_mono.render(m_line1, True, COLOR_TEXT_WHITE)
        metrics_rect_x = metrics_rect.x + 10
        self.screen.blit(t1, (metrics_rect_x, metrics_rect.y + 8))

        best_score_str = f"{self.telemetry.best_score:+.1f}" if self.telemetry.best_score != -math.inf else "N/A"
        time_str = f"{self.telemetry.elapsed_time:.2f}s" if not self.ai_thinking else f"{time.time() - self.telemetry.start_time:.2f}s"
        m_line2 = f"Best Score: {best_score_str}  |  Time: {time_str}  |  Memory Window: {self.agent.memory_window} turns"
        t2 = self.font_mono.render(m_line2, True, COLOR_TEXT_MUTED)
        self.screen.blit(t2, (metrics_rect_x, metrics_rect.y + 30))
        y += 66

        # --- SECTION 2: LIVE SEARCH TREE BRANCH EXPLORATION ---
        sec2_title = self.font_bold.render("SEARCH TREE PATH EXPLORATION", True, COLOR_TEXT_WHITE)
        self.screen.blit(sec2_title, (x, y))
        y += 20

        tree_rect = pygame.Rect(x, y, self.panel_rect.width - 32, 105)
        pygame.draw.rect(self.screen, (20, 24, 32), tree_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_BORDER, tree_rect, 1, border_radius=6)

        ty = tree_rect.y + 8
        curr_branch = self.telemetry.get_current_branch_copy()
        if curr_branch:
            for idx, step_desc in enumerate(curr_branch[:4]):
                prefix = "  " * idx + ("\\-- " if idx > 0 else "Root: ")
                txt = self.font_mono.render(f"{prefix}{step_desc}", True, (100, 200, 255) if idx == 0 else COLOR_TEXT_MUTED)
                self.screen.blit(txt, (tree_rect.x + 8, ty))
                ty += 22
        else:
            txt = self.font_mono.render("Awaiting next AI search invocation...", True, COLOR_TEXT_MUTED)
            self.screen.blit(txt, (tree_rect.x + 8, ty))

        # Show latest pruning cutoff event
        pruned_list = self.telemetry.get_pruned_branches_copy()
        if pruned_list:
            latest_prune = pruned_list[-1]
            p_txt = self.font_small.render(latest_prune, True, COLOR_PRUNED)
            self.screen.blit(p_txt, (tree_rect.x + 8, tree_rect.bottom - 20))

        y += 115

        # --- SECTION 3: CANDIDATE MOVES & SCORES ---
        sec3_title = self.font_bold.render("CANDIDATE MOVES & MINIMAX SCORES", True, COLOR_TEXT_WHITE)
        self.screen.blit(sec3_title, (x, y))
        y += 20

        cand_rect = pygame.Rect(x, y, self.panel_rect.width - 32, 220)
        pygame.draw.rect(self.screen, (20, 24, 32), cand_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_BORDER, cand_rect, 1, border_radius=6)

        cy = cand_rect.y + 8
        items = self.telemetry.get_candidate_scores_copy()
        # Sort by score descending
        items.sort(key=lambda kv: kv[1], reverse=True)

        chosen_str = str(self.telemetry.best_move) if self.telemetry.best_move else ""

        if not items:
            empty_msg = self.font_mono.render("No evaluated moves yet.", True, COLOR_TEXT_MUTED)
            self.screen.blit(empty_msg, (cand_rect.x + 10, cy))
        else:
            for move_str, score in items[:8]:  # Top 8 candidate moves
                is_best = (move_str == chosen_str)
                is_active = (curr_branch and move_str == curr_branch[0])

                if is_best and not self.ai_thinking:
                    prefix = "[CHOSEN] "
                    text_col = COLOR_GREEN_ACCENT
                    bg_col = (25, 55, 35)
                elif is_active:
                    prefix = "[EVAL]   "
                    text_col = (100, 220, 255)
                    bg_col = (30, 45, 65)
                else:
                    prefix = "[CAND]   "
                    text_col = COLOR_TEXT_WHITE
                    bg_col = None

                line_rect = pygame.Rect(cand_rect.x + 4, cy - 2, cand_rect.width - 8, 22)
                if bg_col:
                    pygame.draw.rect(self.screen, bg_col, line_rect, border_radius=4)

                line_text = f"{prefix}{move_str:<32} Score: {score:+.1f}"
                c_surf = self.font_mono.render(line_text, True, text_col)
                self.screen.blit(c_surf, (cand_rect.x + 8, cy))
                cy += 24

        y += 230

        # --- SECTION 4: AI COGNITIVE MEMORY HUD ---
        sec4_title = self.font_bold.render("AI COGNITIVE MEMORY HUD", True, COLOR_TEXT_WHITE)
        self.screen.blit(sec4_title, (x, y))

        mem_win = self.agent.memory_window
        hud_badge = self.font_small.render(f"BUFFER: {mem_win} MOVES  |  BOUNDED HORIZON", True, COLOR_GREEN_ACCENT)
        self.screen.blit(hud_badge, (self.panel_rect.right - hud_badge.get_width() - 20, y + 2))
        y += 20

        mem_rect = pygame.Rect(x, y, self.panel_rect.width - 32, self.panel_rect.bottom - y - 16)
        pygame.draw.rect(self.screen, (20, 24, 32), mem_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_BORDER, mem_rect, 1, border_radius=6)

        my = mem_rect.y + 10

        # Explanatory note confirming tabletop fidelity and bounded rationality
        note1 = self.font_small.render(f"Episodic Memory Horizon: AI retains move provenance for up to {mem_win} turns.", True, (130, 165, 205))
        self.screen.blit(note1, (mem_rect.x + 10, my))
        my += 16
        note2 = self.font_small.render(f"Moves older than {mem_win} turns decay from working memory. Board under-layers are never X-rayed.", True, (100, 130, 160))
        self.screen.blit(note2, (mem_rect.x + 10, my))
        my += 20

        # Capacity gauge / buffer slots
        curr_turn = self.state.turn_number
        recent_moves = self.state.move_history[-mem_win:] if self.state.move_history else []
        slots_used = len(recent_moves)

        slot_w = (mem_rect.width - 20 - (mem_win - 1) * 6) // mem_win
        for s_idx in range(mem_win):
            slot_x = mem_rect.x + 10 + s_idx * (slot_w + 6)
            slot_rect = pygame.Rect(slot_x, my, slot_w, 20)
            is_filled = (s_idx < slots_used)
            s_bg = (30, 52, 42) if is_filled else (26, 30, 40)
            s_border = (46, 204, 113) if is_filled else (45, 52, 68)
            pygame.draw.rect(self.screen, s_bg, slot_rect, border_radius=4)
            pygame.draw.rect(self.screen, s_border, slot_rect, 1, border_radius=4)

            s_text = f"Slot {s_idx + 1}" if not is_filled else f"M-{slots_used - 1 - s_idx}"
            s_surf = self.font_small.render(s_text, True, COLOR_GREEN_ACCENT if is_filled else COLOR_TEXT_MUTED)
            self.screen.blit(s_surf, (slot_rect.centerx - s_surf.get_width() // 2, slot_rect.centery - s_surf.get_height() // 2))

        my += 26

        # Recent action history queue (Last N moves remembered)
        if not recent_moves:
            no_moves = self.font_mono.render(f"Game start: Working memory buffer ready. 0/{mem_win} slots active.", True, COLOR_TEXT_MUTED)
            self.screen.blit(no_moves, (mem_rect.x + 10, my + 15))
        else:
            active_threshold = max(2, mem_win // 2)
            for t_num, mv, desc in reversed(recent_moves):
                turns_ago = curr_turn - t_num
                if turns_ago <= 1:
                    status_col = (100, 220, 255)
                    age_badge = f"[T-{turns_ago} fresh]"
                elif turns_ago <= active_threshold:
                    status_col = COLOR_GREEN_ACCENT
                    age_badge = f"[T-{turns_ago} active]"
                elif turns_ago <= mem_win:
                    status_col = COLOR_TEXT_HIGHLIGHT
                    age_badge = f"[T-{turns_ago} decaying]"
                else:
                    status_col = COLOR_FORGOTTEN
                    age_badge = f"[T-{turns_ago} expired]"

                row_rect = pygame.Rect(mem_rect.x + 8, my, mem_rect.width - 16, 21)
                p_bg = (24, 38, 52) if mv.player == PLAYER_BLUE else (48, 34, 24)
                pygame.draw.rect(self.screen, p_bg, row_rect, border_radius=4)

                m_surf = self.font_mono.render(f"Turn {t_num:<2} {desc:<32}", True, COLOR_TEXT_WHITE)
                self.screen.blit(m_surf, (row_rect.x + 8, row_rect.y + 2))

                stat_surf = self.font_bold.render(age_badge, True, status_col)
                self.screen.blit(stat_surf, (row_rect.right - stat_surf.get_width() - 8, row_rect.y + 2))

                my += 24