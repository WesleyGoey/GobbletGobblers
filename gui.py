"""
gui.py - Pygame GUI with Interactive 5x5 Board, Reserve Trays,
and Live Real-time AI Decision Visualization Panel.

Features:
1. 5x5 Board with charming tactile Gobbler figurine visualization and tabletop fog-of-war.
2. Blue (Human) vs Orange (AI) reserve trays with piece counts and selection highlights.
3. Interactive 2-click move input with valid target destination highlighting.
4. Threaded AI execution without blocking GUI loop:
   - Live search tree branch path display
   - Real-time candidate move scoring
   - Alpha-Beta pruning cutoff logs
   - AI Cognitive Memory HUD (5-move working memory buffer without spoilers)
   - Final chosen move highlighted in vibrant green
5. UI controls: New Game, Goal display (5-in-a-row), AI Difficulty toggle (Easy/Medium/Hard).
"""

import pygame
import threading
import queue
import time
import math
from typing import Tuple, Optional, List, Dict, Any

from game_engine import (
    GameState, Piece, Move,
    PLAYER_BLUE, PLAYER_ORANGE, SIZE_NAMES
)
from ai_agent import MinimaxAgent 
from gui_renderer import BoardRenderer 

# --- Color Palette (Warm Wooden & Slate Tabletop Aesthetic) ---
COLOR_BG = (18, 20, 26)             # Deep slate table surface
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
        "search_budget_ratio": 0.7,
        "time_limit": 0.1,
        "evaluation_noise": 120.0,
        "label": "AI: Easy",
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
        "label": "AI: Medium",
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
        "label": "AI: Hard",
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
    """
    Linearly interpolates between two RGB color tuples.

    Clamps the interpolation factor `t` to the [0.0, 1.0] range to ensure valid RGB channels.
    Used for smooth hover transitions and responsive UI animations.

    Parameters:
        c1: Starting RGB color tuple (r, g, b).
        c2: Target RGB color tuple (r, g, b).
        t: Interpolation factor (0.0 returns c1, 1.0 returns c2).

    Returns:
        Interpolated (r, g, b) integer color tuple.
    """
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


class GobbletGUI:
    def __init__(self, width: int = 760, height: int = 760, win_streak: int = 5, difficulty: str = "medium", show_guide_on_start: bool = True):
        """
        Initializes the Gobblet Gobblers Pygame desktop GUI application.

        Configures:
            - Display surface (width x height), window caption, and 60 FPS clock.
            - Font typography resources for titles, banners, HUD panels, and buttons.
            - Core GameState with fixed 5-in-a-row win condition.
            - Minimax AI agent, SearchTelemetry, and bounded memory parameters.
            - Background AI worker thread orchestration, event queue, and cancellation tokens.
            - UI bounding rectangles for board grid, player reserve trays, and action buttons.
            - Dedicated modular subsystem renderers (BoardRenderer and TelemetryPanelRenderer).

        Parameters:
            width: Screen width in pixels (defaults to 760).
            height: Screen height in pixels (defaults to 760).
            win_streak: Winning line length requirement (must be 5).
            difficulty: AI level ('easy', 'medium', or 'hard').

        Raises:
            ValueError: If win_streak is not 5.
        """
        if win_streak != 5:
            raise ValueError("Only 5-in-a-row is supported as a win condition on a 5x5 board")

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
        self.font_title = pygame.font.SysFont("Helvetica, Arial", 40, bold=True)
        self.font_header = pygame.font.SysFont("Helvetica, Arial", 18, bold=True)
        self.font_bold = pygame.font.SysFont("Helvetica, Arial", 14, bold=True)
        self.font_btn = pygame.font.SysFont("Helvetica, Arial", 12, bold=True)
        self.font_regular = pygame.font.SysFont("Helvetica, Arial", 13)
        self.font_small = pygame.font.SysFont("Helvetica, Arial", 11)
        self.font_mono = pygame.font.SysFont("Courier, Consolas, Menlo", 16, bold=True)

        # Game State (Fixed to 5-in-a-row)
        self.win_streak: int = 5
        self.state = GameState(win_streak=5)
        self.selected_reserve_size: Optional[int] = None
        self.selected_board_pos: Optional[Tuple[int, int]] = None
        self.valid_targets: List[Tuple[int, int]] = []

        # Session Match Timer (5 minutes / 300 seconds, starts upon first placed piece)
        self.session_time_limit: float = 300.0
        self.session_start_time: Optional[float] = None

        # AI Agent & Difficulty Settings
        self.difficulty: str = (
            difficulty.lower() if isinstance(difficulty, str) and difficulty.lower() in DIFFICULTY_CONFIGS else "medium"
        )
        diff_cfg = DIFFICULTY_CONFIGS[self.difficulty]
        self.ai_depth: int = diff_cfg["depth"]
        self.memory_window: int = diff_cfg["memory_window"]
        self.ai_speed_mode = "Normal"  # Kept for backward compatibility
        self.speed_delays = {"Normal": 0.003, "Fast": 0.001, "Instant": 0.0}
        self.agent = MinimaxAgent(
            player=PLAYER_ORANGE,
            max_depth=self.ai_depth,
            memory_window=self.memory_window,
            time_limit=diff_cfg.get("time_limit"),
            search_budget_ratio=diff_cfg.get("search_budget_ratio", 1.0),
            evaluation_noise=diff_cfg.get("evaluation_noise", 0.0),
        )
        self.agent.step_delay = self.speed_delays[self.ai_speed_mode]

        # Threading for background AI computation with search ID generation guard
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

        # Unified Control Bar Buttons: [New Game], [Difficulty], [How to Play]
        btn_w = 152
        gap = 12
        bx = self.board_rect.x
        by = self.board_rect.bottom + 16
        self.btn_new_game = pygame.Rect(bx, by, btn_w, 42)
        self.btn_difficulty = pygame.Rect(bx + btn_w + gap, by, btn_w, 42)
        self.btn_help = pygame.Rect(bx + (btn_w + gap) * 2, by, btn_w, 42)

        # How to Play Guide Modal State & Geometry
        self.show_help_modal: bool = show_guide_on_start
        modal_w = 640
        modal_h = 510
        mx = (self.width - modal_w) // 2
        my = max(10, (self.height - modal_h) // 2)
        self.help_modal_rect = pygame.Rect(mx, my, modal_w, modal_h)
        self.btn_help_close = pygame.Rect(self.help_modal_rect.right - 140, self.help_modal_rect.bottom - 48, 112, 34)
        self.btn_help_x = pygame.Rect(self.help_modal_rect.right - 44, self.help_modal_rect.top + 16, 28, 28)

        # Palette & Configuration Attributes
        self.piece_radii = PIECE_RADII
        self.COLOR_BG = COLOR_BG
        
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

    def set_difficulty(self, difficulty: str):
        """
        Updates the AI difficulty configuration.

        Adjusts Minimax search depth (1, 2, or 3) and cognitive memory window (3, 5, or 8 turns).
        If the AI is currently calculating a move when difficulty changes, cancels the current
        thread search and immediately restarts evaluation under the newly chosen difficulty.

        Parameters:
            difficulty: The requested level ('easy', 'medium', or 'hard').
        """
        diff = (
            difficulty.lower() if isinstance(difficulty, str) and difficulty.lower() in DIFFICULTY_CONFIGS else "medium"
        )
        self.difficulty = diff
        cfg = DIFFICULTY_CONFIGS[diff]
        self.ai_depth = cfg["depth"]
        self.memory_window = cfg["memory_window"]
        self.agent.max_depth = self.ai_depth
        self.agent.memory_window = self.memory_window
        self.agent.time_limit = cfg.get("time_limit")
        self.agent.search_budget_ratio = cfg.get("search_budget_ratio", 1.0)
        self.agent.evaluation_noise = cfg.get("evaluation_noise", 0.0)

        # If AI is currently thinking, restart search immediately with new difficulty
        if self.ai_thinking and not self.state.game_over and self.state.current_player == PLAYER_ORANGE:
            self.start_ai_turn()

    def cycle_difficulty(self):
        """
        Serves as the interactive cycling controller for AI difficulty settings.

        Triggered when the user clicks the 'Difficulty' button in the control panel.
        Transitions the difficulty setting sequentially along the circular preset path:
        `EASY (Depth 1, Memory 3) -> MEDIUM (Depth 2, Memory 5) -> HARD (Depth 3, Memory 8) -> EASY`.
        Calls `set_difficulty()` to immediately synchronize Minimax depth, memory window capacity,
        and cancel/restart active AI background searches.
        """
        curr_idx = DIFFICULTY_LEVELS.index(self.difficulty) if self.difficulty in DIFFICULTY_LEVELS else 1
        next_diff = DIFFICULTY_LEVELS[(curr_idx + 1) % len(DIFFICULTY_LEVELS)]
        self.set_difficulty(next_diff)

    def get_hover(self, key: str, is_hovered: bool, speed: float = 0.3) -> float:
        """
        Computes a responsive smoothed hover factor (0.0 to 1.0) with ease-out interpolation.

        Tracks persistent hover values per UI element key to create fluid animations
        when mouse enters or leaves interactive buttons, cards, and grid cells.

        Parameters:
            key: Unique string identifier for the UI element.
            is_hovered: True if mouse cursor is currently over the element; False otherwise.
            speed: Smoothing lerp rate per frame (default 0.3).

        Returns:
            Current smoothed hover magnitude between 0.0 and 1.0.
        """
        curr = self.hover_anim.get(key, 0.0)
        target = 1.0 if is_hovered else 0.0
        val = curr + (target - curr) * speed
        if abs(val - target) < 0.01:
            val = target
        self.hover_anim[key] = val
        return val

    def check_session_timer(self):
        """Checks if the 5-minute match session has expired; ends the match in a draw if so."""
        if not self.state.game_over and self.session_start_time is not None:
            elapsed = time.time() - self.session_start_time
            if elapsed >= self.session_time_limit:
                self.state.game_over = True
                self.state.winner = None
                self.state.win_reason = "Time limit expired (5:00) - Match ended in a DRAW!"
                self.status_message = "Time limit reached: Draw!"
                if self.cancel_search_event:
                    self.cancel_search_event.set()
                self.ai_thinking = False
                self.pending_ai_move = None

    def run(self):
        """
        Executes the main application loop at a locked 60 FPS.

        Orchestrates event handling, background AI processing, composite drawing,
        and Pygame display buffer flipping until window closure.
        """
        while self.running:
            self.check_session_timer()
            self.handle_events()
            self.process_ai_queue()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    def handle_events(self):
        """
        Pumps and dispatches Pygame window and input events.

        Listens for QUIT events to cleanly terminate the main loop and processes
        left mouse button clicks by routing coordinates to `handle_mouse_click`.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    self.show_help_modal = not self.show_help_modal
                elif event.key == pygame.K_ESCAPE and self.show_help_modal:
                    self.show_help_modal = False
                elif event.key == pygame.K_r:
                    self.reset_game()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos
                self.handle_mouse_click(mouse_pos)

    def handle_mouse_click(self, pos: Tuple[int, int]):
        """
        Routes mouse click coordinates to UI buttons, reserve trays, or board cells.

        Checks interaction gates: ignores board input if the game is over, if it is the AI's turn,
        or if the AI is actively thinking. Dispatches clicks to control buttons (New Game, Goal,
        Difficulty), Blue reserve trays, or board cell coordinates.

        Parameters:
            pos: (x, y) screen pixel coordinates of the click.
        """
        # If guide modal is open, shield all clicks from board/reserves
        if self.show_help_modal:
            if (self.btn_help_close.collidepoint(pos) or
                self.btn_help_x.collidepoint(pos) or
                not self.help_modal_rect.collidepoint(pos)):
                self.show_help_modal = False
            return

        # Check Button Clicks first
        if self.btn_new_game.collidepoint(pos):
            self.reset_game()
            return

        if self.btn_difficulty.collidepoint(pos):
            self.cycle_difficulty()
            return

        if self.btn_help.collidepoint(pos):
            self.show_help_modal = True
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
        """
        Handles user clicks on board cells across all selection states.

        Manages three distinct interaction contexts:
        - Case A: A reserve piece is selected -> Drops the reserve piece onto cell (r, c)
          if valid, or cancels/redirects selection if invalid.
        - Case B: A board piece is locked under Touch-Move -> Repositions the piece to cell (r, c)
          if valid, or enforces touch-move rule if clicking elsewhere.
        - Case C: No piece currently selected -> Attempts to select the friendly Blue piece
          at cell (r, c) via `try_select_board_piece`.

        Parameters:
            r: Target row index (0 to 4).
            c: Target column index (0 to 4).
        """
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
        """
        Attempts to select a board piece with Touch-Move and Uncover Trap enforcement.

        Rules enforced:
        - Uncover Trap check: Tests whether lifting the piece reveals an opponent winning line.
          If revealed and no legal destination can block all opponent lines, Blue suffers an
          instant loss. If blocking moves exist, restricts valid targets strictly to those blocks.
        - Touch-Move rule: Once touched, the piece is locked and must be moved to a valid cell.

        Parameters:
            r: Row index (0 to 4) of selected piece.
            c: Column index (0 to 4) of selected piece.
        """
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
        """
        Populates valid destination cells where a piece of `size` can be dropped from reserve.

        Scans the 5x5 board and includes any cell that is either empty or topped by a piece
        with size strictly less than `size`.

        Parameters:
            size: Size tier of the reserve piece being dropped (1, 2, or 3).
        """
        self.valid_targets = []
        for r in range(5):
            for c in range(5):
                top_size = self.state.get_top_size(r, c)
                if top_size < size:
                    self.valid_targets.append((r, c))

    def calculate_valid_targets_for_move(self, r1: int, c1: int, is_trap: bool = False) -> List[Tuple[int, int]]:
        """
        Calculates all valid destination board cells for moving a piece from (r1, c1).

        Filters cells by size rules (target must be empty or contain a smaller piece).
        If `is_trap` is True, simulates placement on candidate cells and only keeps targets
        that successfully block and break all revealed opponent winning lines.

        Parameters:
            r1: Source row index (0 to 4).
            c1: Source column index (0 to 4).
            is_trap: Whether lifting this piece triggers an Uncover Trap.

        Returns:
            List of valid destination (r2, c2) coordinate tuples.
        """
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
        """
        Executes a validated human move, updates state, and initiates the AI's response turn.

        Applies the move to `self.state`, clears selection indicators, and if the match continues,
        dispatches `start_ai_turn` to compute the AI's counter-move in a background thread.

        Parameters:
            move: The human Move object to execute.
        """
        success = self.state.apply_move(move)
        if success and self.session_start_time is None:
            self.session_start_time = time.time()
        self.selected_reserve_size = None
        self.selected_board_pos = None
        self.valid_targets = []
        self.uncover_trap_warning = False
        self.status_message = ""

        if success and not self.state.game_over:
            # Trigger AI turn in background thread
            self.start_ai_turn()

    def get_remaining_session_time(self) -> float:
        """Returns the remaining seconds in the current 5-minute session."""
        if self.state.game_over:
            return 0.0
        if self.session_start_time is None:
            return self.session_time_limit
        elapsed = time.time() - self.session_start_time
        return max(0.0, self.session_time_limit - elapsed)

    def start_ai_turn(self):
        """
        Launches the Minimax AI calculation asynchronously in a background worker thread.

        Signals cancellation to any prior in-flight searches, increments `search_id` to discard
        stale responses, flushes the event queue, and starts a daemon thread that streams
        AI moves to the GUI without freezing the 60 FPS animation loop.
        """
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
                """
                Worker function executed within the background AI thread.

                Invokes MinimaxAgent.get_action with live step callbacks and pushes
                MOVE_READY to event_queue upon completion.
                """
                def callback(event_type, data):
                    """
                    Serves as the cross-thread event forwarding bridge during Minimax search execution.

                    Invoked synchronously by the Minimax search worker whenever a search event occurs
                    (e.g., node visit, alpha-beta cutoff, root candidate evaluation, or memory update).
                    Checks `current_cancel` to prevent posting stale events from superseded searches,
                    and packs `(current_search_id, event_type, data)` into the thread-safe `event_queue`
                    to be drained by the GUI main thread without causing frame drops or race conditions.

                    Parameters:
                        event_type: Identifier string for the search event ('NODE_VISIT', 'PRUNE', etc.).
                        data: Event payload dictionary containing move data.
                    """
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
        """
        Processes pending moves delivered by the AI thread.

        Drains events from `event_queue`, filters out stale events from previous searches,
        captures MOVE_READY, and enforces a brief visual highlight pause (0.4s) before
        applying the chosen AI move to the board so the user can see the decision clearly.
        """
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
        """
        Resets the board state, cancels running AI threads, and initializes a new game.

        Parameters:
            win_streak: Optional new win goal (must be 5). Preserves current goal if None.

        Raises:
            ValueError: If win_streak is specified and not 5.
        """
        if win_streak is not None and win_streak != 5:
            raise ValueError("Only 5-in-a-row is supported as a win condition on a 5x5 board")
        if self.cancel_search_event:
            self.cancel_search_event.set()
        self.search_id += 1  # Invalidate any running background search
        self.win_streak = 5
        self.state = GameState(win_streak=5)
        self.session_start_time = None
        self.selected_reserve_size = None
        self.selected_board_pos = None
        self.valid_targets = []
        self.uncover_trap_warning = False
        self.status_message = ""
        cfg = DIFFICULTY_CONFIGS[self.difficulty]
        self.agent.max_depth = self.ai_depth
        self.agent.memory_window = self.memory_window
        self.agent.time_limit = cfg.get("time_limit")
        self.agent.search_budget_ratio = cfg.get("search_budget_ratio", 1.0)
        self.agent.evaluation_noise = cfg.get("evaluation_noise", 0.0)
        self.ai_thinking = False
        self.pending_ai_move = None
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break

    def get_cell_rect(self, r: int, c: int) -> pygame.Rect:
        """
        Calculates the on-screen pixel bounding box (pygame.Rect) for grid cell (r, c).

        Parameters:
            r: Row index (0 to 4).
            c: Column index (0 to 4).

        Returns:
            A pygame.Rect with pixel coordinates and dimensions matching the cell tile.
        """
        x = self.board_rect.x + c * (self.cell_size + self.cell_gap) + 4
        y = self.board_rect.y + r * (self.cell_size + self.cell_gap) + 4
        return pygame.Rect(x, y, self.cell_size, self.cell_size)

    # ==================== DRAWING ROUTINES & SUBSYSTEM DELEGATION ====================

    def draw(self):
        """
        Renders the complete GUI frame across all visual layers.

        Draws background fill, header banner, reserve trays, board grid and pieces,
        and control buttons.
        """
        self.screen.fill(COLOR_BG)
        self.draw_header()
        self.draw_reserves()
        self.draw_board()
        self.draw_controls()
        if self.show_help_modal:
            self.renderer.draw_help_modal()

    def draw_header(self):
        """
        Serves as the facade rendering delegator for the application header banner.

        Delegates rendering to `BoardRenderer.draw_header()`, passing GUI state to draw
        the game title, subtitle (grid size, win streak, difficulty mode, search depth),
        and the dynamic active-turn status card.
        """
        self.renderer.draw_header()

    def draw_reserves(self):
        """
        Serves as the facade rendering delegator for the side reserve piece trays.

        Delegates rendering to `BoardRenderer.draw_reserves()`, rendering Blue's tray
        on the left and Orange's tray on the right, complete with piece counts, 3D figurines,
        selection highlights, and depleted slot indicators.
        """
        self.renderer.draw_reserves()

    def draw_board(self):
        """
        Serves as the facade rendering delegator for the primary 5x5 game board.

        Delegates rendering to `BoardRenderer.draw_board()`, drawing board wooden styling,
        grid grooves, cell bevels, legal target placement markers, mouse hover highlights,
        and visible top pieces. Also triggers victory celebration effects when game is won.
        """
        self.renderer.draw_board()

    def draw_winning_beam(self, winning_line: List[Tuple[int, int]]):
        """
        Serves as the facade delegator for the victory connection beam effect.

        Delegates to `BoardRenderer.draw_winning_beam()`, rendering a multi-layered glowing
        laser beam and connector halos along the straight-line coordinates of the winning line.

        Parameters:
            winning_line: List of consecutive (r, c) cell coordinates forming the winning 5-in-a-row.
        """
        self.renderer.draw_winning_beam(winning_line)

    def draw_winning_piece_auras(self, winning_line: List[Tuple[int, int]]):
        """
        Serves as the facade delegator for victory piece crown auras.

        Delegates to `BoardRenderer.draw_winning_piece_auras()`, drawing pulsing golden rings
        and sparkling victory diamond stars directly on top of pieces in the winning line.

        Parameters:
            winning_line: List of (r, c) cell coordinates belonging to the winning line.
        """
        self.renderer.draw_winning_piece_auras(winning_line)

    def draw_winning_celebration(self, winning_line: List[Tuple[int, int]]):
        """
        Serves as the composite victory celebration effect delegator.

        Delegates to `BoardRenderer.draw_winning_celebration()`, coordinating both the floor-level
        glowing connection beam and the top-level golden piece crown auras.

        Parameters:
            winning_line: List of (r, c) cell coordinates forming the winning alignment.
        """
        self.renderer.draw_winning_celebration(winning_line)

    def draw_cell_stack(self, center: Tuple[int, int], stack: List[Piece]):
        """
        Serves as the facade delegator for drawing a cell's piece stack.

        Delegates to `BoardRenderer.draw_cell_stack()`, rendering the topmost visible piece
        at screen pixel center `center`, or skipping rendering if the stack is vacant.

        Parameters:
            center: Pixel coordinates (x, y) on the screen canvas representing the cell center.
            stack: List of Piece objects occupying the cell from base to top.
        """
        self.renderer.draw_cell_stack(center, stack)

    def draw_piece_3d(self, center: Tuple[int, int], size: int, player: int, is_top: bool = True, is_disabled: bool = False):
        """
        Serves as the tactile 3D Gobbler figurine rendering delegator.

        Delegates to `BoardRenderer.draw_piece_3d()`, fetching pre-rendered flyweight sprites
        from `PieceSpriteCache` and blitting them with sub-pixel centering onto the screen surface.

        Parameters:
            center: Screen pixel coordinate (x, y) where the piece figurine should be centered.
            size: Piece size tier (1 = Small, 2 = Medium, 3 = Large).
            player: Player identifier (PLAYER_BLUE or PLAYER_ORANGE) determining color palette.
            is_top: True if piece is topmost/visible; False if partially occluded.
            is_disabled: True if piece slot is depleted or unavailable for interaction.
        """
        self.renderer.draw_piece_3d(center, size, player, is_top=is_top, is_disabled=is_disabled)

    def draw_controls(self):
        """
        Serves as the facade delegator for the bottom control bar.

        Delegates to `BoardRenderer.draw_controls()`, rendering interactive UI buttons
        ('Difficulty', 'New Game / Reset', 'Goal') with smooth hover animations and active status texts.
        """
        self.renderer.draw_controls()
