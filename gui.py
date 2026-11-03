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
from gui_renderer import BoardRenderer, lerp_color
from gui_telemetry import TelemetryPanelRenderer

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

        # Palette & Configuration Attributes
        self.piece_radii = PIECE_RADII
        self.COLOR_BG = COLOR_BG
        self.COLOR_PANEL_BG = COLOR_PANEL_BG
        self.COLOR_CARD_BG = COLOR_CARD_BG
        self.COLOR_BORDER = COLOR_BORDER
        self.COLOR_BOARD_BG = COLOR_BOARD_BG
        self.COLOR_BOARD_BORDER = COLOR_BOARD_BORDER
        self.COLOR_CELL_BG = COLOR_CELL_BG
        self.COLOR_CELL_LIGHT = COLOR_CELL_LIGHT
        self.COLOR_CELL_SHADOW = COLOR_CELL_SHADOW
        self.COLOR_CELL_HOVER = COLOR_CELL_HOVER
        self.COLOR_VALID_TARGET = COLOR_VALID_TARGET
        self.COLOR_VALID_GLOW = COLOR_VALID_GLOW
        self.COLOR_SELECTED_GLOW = COLOR_SELECTED_GLOW
        self.COLOR_WINNING_CELL = COLOR_WINNING_CELL
        self.COLOR_BLUE = COLOR_BLUE
        self.COLOR_BLUE_LIGHT = COLOR_BLUE_LIGHT
        self.COLOR_BLUE_DARK = COLOR_BLUE_DARK
        self.COLOR_ORANGE = COLOR_ORANGE
        self.COLOR_ORANGE_LIGHT = COLOR_ORANGE_LIGHT
        self.COLOR_ORANGE_DARK = COLOR_ORANGE_DARK
        self.COLOR_TEXT_WHITE = COLOR_TEXT_WHITE
        self.COLOR_TEXT_MUTED = COLOR_TEXT_MUTED
        self.COLOR_TEXT_HIGHLIGHT = COLOR_TEXT_HIGHLIGHT
        self.COLOR_GREEN_ACCENT = COLOR_GREEN_ACCENT
        self.COLOR_CORAL_ACCENT = COLOR_CORAL_ACCENT
        self.COLOR_RED_ACCENT = COLOR_RED_ACCENT
        self.COLOR_PRUNED = COLOR_PRUNED
        self.COLOR_REMEMBERED = COLOR_REMEMBERED
        self.COLOR_FORGOTTEN = COLOR_FORGOTTEN
        self.DIFFICULTY_CONFIGS = DIFFICULTY_CONFIGS

        # Modular Renderer & Telemetry Subsystems
        self.renderer = BoardRenderer(self)
        self.telemetry_renderer = TelemetryPanelRenderer(self)

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

    # ==================== DRAWING ROUTINES & SUBSYSTEM DELEGATION ====================

    def draw(self):
        self.screen.fill(COLOR_BG)
        self.draw_header()
        self.draw_reserves()
        self.draw_board()
        self.draw_controls()
        self.draw_visualization_panel()

    def draw_header(self):
        self.renderer.draw_header()

    def draw_reserves(self):
        self.renderer.draw_reserves()

    def draw_board(self):
        self.renderer.draw_board()

    def draw_winning_beam(self, winning_line: List[Tuple[int, int]]):
        self.renderer.draw_winning_beam(winning_line)

    def draw_winning_piece_auras(self, winning_line: List[Tuple[int, int]]):
        self.renderer.draw_winning_piece_auras(winning_line)

    def draw_winning_celebration(self, winning_line: List[Tuple[int, int]]):
        self.renderer.draw_winning_celebration(winning_line)

    def draw_cell_stack(self, center: Tuple[int, int], stack: List[Piece]):
        self.renderer.draw_cell_stack(center, stack)

    def draw_piece_3d(self, center: Tuple[int, int], size: int, player: int, is_top: bool = True, is_disabled: bool = False):
        self.renderer.draw_piece_3d(center, size, player, is_top=is_top, is_disabled=is_disabled)

    def draw_controls(self):
        self.renderer.draw_controls()

    def draw_visualization_panel(self):
        self.telemetry_renderer.draw_visualization_panel()
