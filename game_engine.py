"""
game_engine.py - Core game state, rules, move validation, and win checking
for 5x5 Gobblet Gobblers.

Specifications:
- 5x5 board (25 cells)
- Each cell is a LIFO stack (up to 3 nested pieces, strictly descending size: largest on top)
- 2 Players: Player 1 (Blue / Human), Player 2 (Orange / AI)
- 12 pieces per player: 4 Large (size 3), 4 Medium (size 2), 4 Small (size 1)
- Win condition: 5-in-a-row of visible pieces
- Touch-move & Uncover Trap rule support
"""

from typing import List, Tuple, Optional, Dict, NamedTuple

PLAYER_BLUE = 1    # Human
PLAYER_ORANGE = 2  # AI

PLAYER_NAMES = {
    PLAYER_BLUE: "Blue (Human)",
    PLAYER_ORANGE: "Orange (AI)"
}

SIZE_NAMES = {
    1: "Small (S)",
    2: "Medium (M)",
    3: "Large (L)"
}

SIZE_SYMBOLS = {
    1: "S",
    2: "M",
    3: "L"
}


class Piece(NamedTuple):
    player: int  # PLAYER_BLUE (1) or PLAYER_ORANGE (2)
    size: int    # 1, 2, or 3

    def __str__(self) -> str:
        """
        Returns a concise human-readable 2-character code representing the piece.

        Format:
            - First character: Player identifier ('B' for Blue/Human, 'O' for Orange/AI).
            - Second character: Piece size symbol ('S' for Small, 'M' for Medium, 'L' for Large).
        Example:
            'BL' denotes Blue Large, 'OS' denotes Orange Small.
        """
        color = "B" if self.player == PLAYER_BLUE else "O"
        return f"{color}{SIZE_SYMBOLS[self.size]}"


class Move:
    """Represents a game action: either a reserve drop or a board reposition."""
    def __init__(
        self,
        player: int,
        move_type: str,  # 'drop' or 'move'
        to_pos: Tuple[int, int],
        size: Optional[int] = None,
        from_pos: Optional[Tuple[int, int]] = None
    ):
        """
        Initializes a Move instance representing an action in the Gobblet Gobblers game.

        Parameters:
            player: The player initiating the move (PLAYER_BLUE or PLAYER_ORANGE).
            move_type: The category of action ('drop' for introducing a reserve piece,
                       or 'move' for repositioning an existing piece on the board).
            to_pos: Target (row, column) board coordinate (0-4, 0-4) where the piece lands.
            size: The size tier of the piece being moved (1 = Small, 2 = Medium, 3 = Large).
                  Mandatory for 'drop' moves; automatically resolved from board for 'move'.
            from_pos: Source (row, column) board coordinate when move_type == 'move'.
                      Must be None for 'drop' actions.

        Attributes:
            uncovered_piece: Stores the piece revealed beneath from_pos (if any) when lifted.
            covered_piece: Stores the piece buried underneath to_pos (if any) upon landing.
        """
        self.player = player
        self.move_type = move_type
        self.to_pos = to_pos
        self.size = size
        self.from_pos = from_pos
        # Captured metadata for undo / history tracking
        self.uncovered_piece: Optional[Piece] = None
        self.covered_piece: Optional[Piece] = None

    def __str__(self) -> str:
        """
        Formats the Move into an intuitive descriptive string for logs and UI display.

        Example outputs:
            - '[Blue] Drop L -> (2,2)'
            - '[Orange] Move S3 (1,1) -> (2,2)'
        """
        r2, c2 = self.to_pos
        p_name = "Blue" if self.player == PLAYER_BLUE else "Orange"
        if self.move_type == 'drop':
            return f"[{p_name}] Drop {SIZE_SYMBOLS[self.size]} -> ({r2},{c2})"
        else:
            r1, c1 = self.from_pos
            s_str = f"S{self.size}" if self.size else "Piece"
            return f"[{p_name}] Move {s_str} ({r1},{c1}) -> ({r2},{c2})"

    def __repr__(self) -> str:
        """
        Serves as the canonical string representation of the Move instance.

        Primarily used for developer debugging, REPL inspection, log trace formatting,
        and test assertion diagnostic messages by matching the descriptive output of `__str__()`.
        """
        return self.__str__()

    def __eq__(self, other) -> bool:
        """
        Determines structural equality between two Move instances.

        Returns True if both moves share identical player, move_type, target position,
        size, and source position; False otherwise or if other is not a Move.
        """
        if not isinstance(other, Move):
            return False
        return (
            self.player == other.player and
            self.move_type == other.move_type and
            self.to_pos == other.to_pos and
            self.size == other.size and
            self.from_pos == other.from_pos
        )

    def __hash__(self) -> int:
        """
        Serves as the hash code generator for Move instances.

        Computes a deterministic integer hash value based on the immutable core attributes:
        `(player, move_type, to_pos, size, from_pos)`. Enables Move objects to serve as keys
        in dictionaries, membership sets, transposition tables, and telemetry lookup structures.
        """
        return hash((self.player, self.move_type, self.to_pos, self.size, self.from_pos))


def generate_winning_lines(win_streak: int = 5) -> List[List[Tuple[int, int]]]:
    """
    Precomputes all possible contiguous winning line segments on a 5x5 board.

    Scans horizontal rows, vertical columns, main diagonals (top-left to bottom-right),
    and anti-diagonals (top-right to bottom-left) to extract straight line coordinates
    of length `win_streak`.

    Parameters:
        win_streak: Target consecutive alignment length (5-in-a-row generates 12 lines:
                    5 horizontal + 5 vertical + 2 diagonal).

    Returns:
        A list of coordinate lists, where each inner list contains `win_streak` (row, col) tuples.

    Raises:
        ValueError: If win_streak is not 5.
    """
    if win_streak != 5:
        raise ValueError("Only 5-in-a-row is supported as a win condition on a 5x5 board")

    lines = []
    # Horizontal lines
    for r in range(5):
        for c in range(5 - win_streak + 1):
            lines.append([(r, c + i) for i in range(win_streak)])

    # Vertical lines
    for c in range(5):
        for r in range(5 - win_streak + 1):
            lines.append([(r + i, c) for i in range(win_streak)])

    # Main diagonals (top-left to bottom-right, delta=(1,1))
    for r in range(5 - win_streak + 1):
        for c in range(5 - win_streak + 1):
            lines.append([(r + i, c + i) for i in range(win_streak)])

    # Anti-diagonals (top-right to bottom-left, delta=(1,-1))
    for r in range(5 - win_streak + 1):
        for c in range(win_streak - 1, 5):
            lines.append([(r + i, c - i) for i in range(win_streak)])

    return lines


# Precomputed cached line configurations for standard 5x5 board (12 lines for 5-in-a-row)
WINNING_LINES_5 = generate_winning_lines(5)
WINNING_LINES = WINNING_LINES_5

# Precomputed cell-to-lines index for O(1) lookup of lines intersecting any cell (r, c)
CELL_LINES_5: Dict[Tuple[int, int], List[List[Tuple[int, int]]]] = {
    (r, c): [line for line in WINNING_LINES_5 if (r, c) in line]
    for r in range(5) for c in range(5)
}
CELL_LINES = CELL_LINES_5


class GameState:
    """Full representation of a 5x5 Gobblet Gobblers game state."""

    def __init__(self, win_streak: int = 5):
        """
        Initializes a fresh 5x5 Gobblet Gobblers game state.

        Sets up:
            - The winning line criteria (5-in-a-row only).
            - Precomputed winning lines and fast O(1) cell-to-intersecting-lines lookup dictionaries.
            - 5x5 board grid where each cell is a LIFO stack (list) of Piece objects.
            - Reserve inventory for each player: 4 Large (size 3), 4 Medium (size 2), 4 Small (size 1).
            - Turn sequence starting with PLAYER_BLUE (turn_number = 1).
            - Victory tracking attributes (winner, game_over flag, win_reason, winning_line coordinates).
            - Move history log and covered turns temporal tracking for AI memory bounded rationality.

        Parameters:
            win_streak: Alignment length required to win (must be 5).
        """
        if win_streak != 5:
            raise ValueError("Only 5-in-a-row counts as a win condition on a 5x5 board")
        self.win_streak: int = 5
        self.winning_lines = WINNING_LINES_5
        self.cell_lines = CELL_LINES_5

        # 5x5 board, each cell is a LIFO list of Piece objects.
        # Bottom piece at index 0, top piece at index -1.
        self.board: List[List[List[Piece]]] = [[[] for _ in range(5)] for _ in range(5)]

        # Reserves: 4 Large (3), 4 Medium (2), 4 Small (1) per player
        self.reserves: Dict[int, Dict[int, int]] = {
            PLAYER_BLUE: {1: 4, 2: 4, 3: 4},
            PLAYER_ORANGE: {1: 4, 2: 4, 3: 4}
        }

        self.current_player: int = PLAYER_BLUE
        self.turn_number: int = 1
        self.winner: Optional[int] = None
        self.game_over: bool = False
        self.win_reason: str = ""
        self.winning_line: Optional[List[Tuple[int, int]]] = None

        # Move history: list of tuples (turn_number, Move, description)
        self.move_history: List[Tuple[int, Move, str]] = []

        # Tracks at what turn each piece in each cell stack was covered (None if top/uncovered)
        self.covered_turns: List[List[List[Optional[int]]]] = [[[] for _ in range(5)] for _ in range(5)]

    def get_top_piece(self, r: int, c: int) -> Optional[Piece]:
        """
        Retrieves the topmost (visible) Piece at the specified board cell.

        Parameters:
            r: Row index (0 to 4).
            c: Column index (0 to 4).

        Returns:
            The visible Piece object at cell (r, c), or None if the cell is unoccupied.
        """
        cell = self.board[r][c]
        return cell[-1] if cell else None

    def get_top_size(self, r: int, c: int) -> int:
        """
        Returns the size of the topmost piece at cell (r, c).

        Returns:
            1, 2, or 3 for Small, Medium, or Large pieces respectively; returns 0 if empty.
        """
        piece = self.get_top_piece(r, c)
        return piece.size if piece else 0

    def get_stack_depth(self, r: int, c: int) -> int:
        """
        Returns the total number of nested pieces stacked at cell (r, c).

        Returns:
            An integer between 0 (empty) and 3 (fully nested stack).
        """
        return len(self.board[r][c])

    def get_cell_stack(self, r: int, c: int) -> List[Piece]:
        """
        Serves as the piece stack inspector for cell (r, c).

        Returns a shallow copy list of all nested Piece objects residing at the specified board cell,
        ordered sequentially from the base piece (index 0) up to the topmost visible piece (index -1).
        Returning a defensive copy guarantees that caller inspection (e.g. GUI rendering layers,
        AI belief state generation, or test assertions) cannot accidentally mutate internal board stacks.

        Parameters:
            r: Row index (0 to 4).
            c: Column index (0 to 4).

        Returns:
            A list of Piece instances currently stacked at (r, c). Returns empty list if cell is vacant.
        """
        return list(self.board[r][c])

    def get_cell_covered_turns(self, r: int, c: int) -> List[Optional[int]]:
        """
        Returns the list of turn numbers when each piece at cell (r, c) was buried.

        For buried pieces, the value is an integer turn number indicating when another
        piece was placed on top of it. For the top uncovered piece, the value is None.
        Used by the bounded rationality forgetting memory mechanism.
        """
        return list(self.covered_turns[r][c])

    def set_cell(
        self,
        r: int,
        c: int,
        pieces: List[Piece],
        covered_turns: Optional[List[Optional[int]]] = None
    ):
        """
        Configures the piece stack and covered turns history for a specific cell.

        Serves primarily as a test fixture and tactical scenario builder to directly inject
        custom piece stacks and simulated temporal memory states without playing through
        all preceding turns.

        Parameters:
            r: Row index (0 to 4).
            c: Column index (0 to 4).
            pieces: List of Piece objects ordered from bottom (index 0) to top (index -1).
            covered_turns: Optional list of turn numbers matching `pieces`. If omitted,
                           defaults to assigning turn 1 for buried layers and None for the top.
        """
        self.board[r][c] = list(pieces)
        if covered_turns is not None:
            self.covered_turns[r][c] = list(covered_turns)
        else:
            n = len(pieces)
            if n == 0:
                self.covered_turns[r][c] = []
            elif n == 1:
                self.covered_turns[r][c] = [None]
            else:
                self.covered_turns[r][c] = [1] * (n - 1) + [None]

    def clone(self) -> 'GameState':
        """
        Creates a fast deep clone of the current GameState.

        Optimized for game tree search simulations (Minimax / Alpha-Beta). Clones board stacks,
        reserves, turn counters, and tracking metadata while sharing immutable precomputed
        winning line tables to maximize branching performance.

        Returns:
            A distinct, fully independent GameState instance.
        """
        new_state = GameState.__new__(GameState)
        new_state.win_streak = 5
        new_state.winning_lines = self.winning_lines
        new_state.cell_lines = getattr(self, 'cell_lines', CELL_LINES_5)
        new_state.board = [[cell.copy() for cell in row] for row in self.board]
        new_state.covered_turns = [[cell.copy() for cell in row] for row in self.covered_turns]
        new_state.reserves = {
            PLAYER_BLUE: self.reserves[PLAYER_BLUE].copy(),
            PLAYER_ORANGE: self.reserves[PLAYER_ORANGE].copy()
        }
        new_state.current_player = self.current_player
        new_state.turn_number = self.turn_number
        new_state.winner = self.winner
        new_state.game_over = self.game_over
        new_state.win_reason = self.win_reason
        new_state.winning_line = list(self.winning_line) if self.winning_line else None
        new_state.move_history = self.move_history.copy()
        return new_state

    def find_all_winning_lines(self) -> Dict[int, List[List[Tuple[int, int]]]]:
        """
        Scans all precomputed winning lines on the 5x5 board for completed monochromatic streaks.

        Only visible (topmost) pieces are evaluated. Examines horizontal, vertical, and diagonal
        alignments according to self.win_streak.

        Returns:
            A dictionary mapping player ID (PLAYER_BLUE / PLAYER_ORANGE) to a list of completed
            winning line coordinate paths.
        """
        wins: Dict[int, List[List[Tuple[int, int]]]] = {PLAYER_BLUE: [], PLAYER_ORANGE: []}
        board = self.board
        for line in self.winning_lines:
            c0 = board[line[0][0]][line[0][1]]
            if not c0:
                continue
            p = c0[-1].player
            match = True
            for r, c in line[1:]:
                cell = board[r][c]
                if not cell or cell[-1].player != p:
                    match = False
                    break
            if match:
                wins[p].append(line)
        return wins

    def check_winner_on_board(self) -> Tuple[Optional[int], Optional[List[Tuple[int, int]]]]:
        """
        Determines if any player has achieved a winning alignment on the visible board.

        Evaluates completed lines for both players. In the rare edge case where both players
        have simultaneously completed lines (e.g., triggered by an uncover action), rule
        priority is awarded to the opponent of the player whose turn it currently is.

        Returns:
            A tuple of (winner_player_id, winning_line_coordinates), or (None, None) if no win.
        """
        wins = self.find_all_winning_lines()
        blue_wins = wins[PLAYER_BLUE]
        orange_wins = wins[PLAYER_ORANGE]

        if blue_wins and not orange_wins:
            return PLAYER_BLUE, blue_wins[0]
        if orange_wins and not blue_wins:
            return PLAYER_ORANGE, orange_wins[0]
        if blue_wins and orange_wins:
            opp = self.get_opponent(self.current_player)
            return opp, wins[opp][0]
        return None, None

    def get_opponent(self, player: int) -> int:
        """
        Returns the opposing player identifier for the given player.

        Parameters:
            player: Either PLAYER_BLUE (1) or PLAYER_ORANGE (2).

        Returns:
            PLAYER_ORANGE if player is PLAYER_BLUE; PLAYER_BLUE if player is PLAYER_ORANGE.
        """
        return PLAYER_ORANGE if player == PLAYER_BLUE else PLAYER_BLUE

    def would_lift_uncover_opponent_win(self, from_pos: Tuple[int, int]) -> Tuple[bool, List[List[Tuple[int, int]]]]:
        """
        Simulates lifting the top piece at from_pos to evaluate the Uncover Trap rule.

        Temporarily pops the top piece at from_pos and checks if doing so reveals an underlying
        opponent piece that completes one or more winning lines for the opponent.

        Parameters:
            from_pos: Board coordinate (r, c) of the piece considered for lifting.

        Returns:
            A tuple (is_uncover_trap, opponent_winning_lines):
            - is_uncover_trap: True if lifting reveals an opponent win; False otherwise.
            - opponent_winning_lines: List of winning line coordinate paths revealed for the opponent.
        """
        r, c = from_pos
        cell = self.board[r][c]
        if len(cell) < 2:
            return False, []

        piece = cell[-1]
        opponent = self.get_opponent(piece.player)
        if cell[-2].player != opponent:
            return False, []

        cell.pop()
        opp_lines = []
        cell_lines = getattr(self, 'cell_lines', None)
        relevant_lines = cell_lines[(r, c)] if cell_lines else [line for line in self.winning_lines if (r, c) in line]
        
        board = self.board
        win_streak = self.win_streak
        for line in relevant_lines:
            colors_count = 0
            for lr, lc in line:
                lcell = board[lr][lc]
                if lcell and lcell[-1].player == opponent:
                    colors_count += 1
                else:
                    break
            if colors_count == win_streak:
                opp_lines.append(line)

        cell.append(piece)  # Restore piece
        return (len(opp_lines) > 0), opp_lines

    def get_legal_moves(self, player: Optional[int] = None) -> List[Move]:
        """
        Generates the comprehensive list of all legal moves available to a player.

        Enforces all standard Gobblet Gobblers rules:
        1. Action A (Reserve Drops): Introduce an available piece from reserves onto an empty
           cell or onto any cell occupied by a strictly smaller piece (gobbling).
        2. Action B (Board Moves): Reposition any top piece belonging to the player to a different
           cell that is empty or contains a strictly smaller piece.
        3. Uncover Trap Enforcement: If lifting a board piece reveals an opponent's winning line,
           that piece MUST be relocated to a cell that blocks and breaks ALL revealed opponent lines.
           If no destination can block all lines, moving that piece is illegal (or results in instant loss).

        Parameters:
            player: The player whose moves to generate (defaults to current_player).

        Returns:
            A list of valid Move objects. Returns empty list if game_over is True.
        """
        if self.game_over:
            return []

        if player is None:
            player = self.current_player

        opponent = self.get_opponent(player)
        legal_moves: List[Move] = []
        board = self.board

        # Fast pre-extraction of top piece sizes across the 5x5 board
        top_sizes = [
            [board[r][c][-1].size if board[r][c] else 0 for c in range(5)]
            for r in range(5)
        ]

        # 1. Action A: Introduce from Reserve
        reserve = self.reserves[player]
        for size in (3, 2, 1):  # Check largest first for optimal ordering
            if reserve[size] > 0:
                for r in range(5):
                    row_sizes = top_sizes[r]
                    for c in range(5):
                        if row_sizes[c] < size:
                            legal_moves.append(Move(
                                player=player,
                                move_type='drop',
                                size=size,
                                to_pos=(r, c)
                            ))

        # 2. Action B: Move on Board
        board_moves: List[Move] = []
        for r1 in range(5):
            row_board = board[r1]
            for c1 in range(5):
                cell = row_board[c1]
                if cell and cell[-1].player == player:
                    size = cell[-1].size
                    is_trap, opp_lines = self.would_lift_uncover_opponent_win((r1, c1))

                    for r2 in range(5):
                        row_sizes2 = top_sizes[r2]
                        for c2 in range(5):
                            if (r1, c1) == (r2, c2):
                                continue
                            if row_sizes2[c2] < size:
                                move = Move(
                                    player=player,
                                    move_type='move',
                                    size=size,
                                    from_pos=(r1, c1),
                                    to_pos=(r2, c2)
                                )
                                if is_trap:
                                    # If lifting created an opponent win, this move MUST block all revealed opponent lines!
                                    p = cell.pop()
                                    board[r2][c2].append(p)
                                    opp_wins = self.find_all_winning_lines()[opponent]
                                    board[r2][c2].pop()
                                    cell.append(p)

                                    if len(opp_wins) == 0:
                                        board_moves.append(move)
                                else:
                                    board_moves.append(move)

        legal_moves.extend(board_moves)
        return legal_moves

    def apply_move(self, move: Move) -> bool:
        """
        Executes a legal move, updating board state, reserves, and game status.

        Process:
        1. Verifies turn order and legality of drop/move parameters.
        2. Executes piece placement or board relocation, updating LIFO cell stacks and covered turns.
        3. Evaluates Uncover Trap consequences: if lifting revealed an opponent winning line
           and the destination failed to block it, active player immediately forfeits to opponent.
        4. Detects win streaks for both players, prioritizing the active player (or opponent if uncover).
        5. If no win occurred, calls `_advance_turn(move)` to toggle player and check immobility.

        Parameters:
            move: The Move instance to apply.

        Returns:
            True if the move was successfully validated and applied; False if the move was illegal.
        """
        if self.game_over:
            return False

        # Strictly enforce turn order
        if move.player != self.current_player:
            return False

        player = move.player
        opponent = self.get_opponent(player)

        if move.move_type == 'drop':
            # Validation
            size = move.size
            if self.reserves[player][size] <= 0:
                return False
            r, c = move.to_pos
            if self.get_top_size(r, c) >= size:
                return False

            # Execute drop
            self.reserves[player][size] -= 1
            piece = Piece(player, size)
            if self.board[r][c]:
                move.covered_piece = self.board[r][c][-1]
                if self.covered_turns[r][c]:
                    self.covered_turns[r][c][-1] = self.turn_number
            self.board[r][c].append(piece)
            self.covered_turns[r][c].append(None)

            # Check win condition for player
            wins = self.find_all_winning_lines()
            if wins[player]:
                self.winner = player
                self.winning_line = wins[player][0]
                self.game_over = True
                self.win_reason = f"{PLAYER_NAMES[player]} completed {self.win_streak}-in-a-row!"
            else:
                self._advance_turn(move)
            return True

        elif move.move_type == 'move':
            r1, c1 = move.from_pos
            r2, c2 = move.to_pos
            top_piece = self.get_top_piece(r1, c1)
            if not top_piece or top_piece.player != player:
                return False
            size = top_piece.size
            if (r1, c1) == (r2, c2):
                return False
            if self.get_top_size(r2, c2) >= size:
                return False

            # Step 1: Lift piece
            lifted_piece = self.board[r1][c1].pop()
            if self.covered_turns[r1][c1]:
                self.covered_turns[r1][c1].pop()
                if self.covered_turns[r1][c1]:
                    self.covered_turns[r1][c1][-1] = None  # Now top and uncovered
            move.uncovered_piece = self.get_top_piece(r1, c1)

            # Step 2: Uncover Trap Check (Did lifting reveal opponent's winning line?)
            lift_wins = self.find_all_winning_lines()
            uncovered_opponent_win = len(lift_wins[opponent]) > 0

            # Step 3: Place piece onto destination
            if self.board[r2][c2]:
                move.covered_piece = self.board[r2][c2][-1]
                if self.covered_turns[r2][c2]:
                    self.covered_turns[r2][c2][-1] = self.turn_number
            self.board[r2][c2].append(lifted_piece)
            self.covered_turns[r2][c2].append(None)

            # Step 4: Evaluate post-placement outcome
            final_wins = self.find_all_winning_lines()
            opp_has_win = len(final_wins[opponent]) > 0
            player_has_win = len(final_wins[player]) > 0

            if uncovered_opponent_win and opp_has_win:
                # Active player walked into Uncover Trap and failed to block!
                self.winner = opponent
                self.winning_line = final_wins[opponent][0]
                self.game_over = True
                self.win_reason = f"Uncover Trap! {PLAYER_NAMES[player]} revealed opponent winning line and couldn't block it!"
                return True

            if player_has_win:
                self.winner = player
                self.winning_line = final_wins[player][0]
                self.game_over = True
                self.win_reason = f"{PLAYER_NAMES[player]} completed {self.win_streak}-in-a-row!"
                return True

            if opp_has_win:
                self.winner = opponent
                self.winning_line = final_wins[opponent][0]
                self.game_over = True
                self.win_reason = f"{PLAYER_NAMES[opponent]} completed {self.win_streak}-in-a-row!"
                return True

            self._advance_turn(move)
            return True

        return False

    def _advance_turn(self, move: Move):
        """
        Finalizes the active turn, updates history, and transitions to the next player.

        Appends the move to `move_history`, increments `turn_number`, switches `current_player`,
        and checks whether the next player has any legal moves available. If the incoming player
        is immobilized (has zero legal moves), the game terminates with an immobility loss.

        Parameters:
            move: The Move that was just executed.
        """
        desc = str(move)
        self.move_history.append((self.turn_number, move, desc))
        self.turn_number += 1
        self.current_player = self.get_opponent(self.current_player)

        # Check if next player has any legal moves available
        next_moves = self.get_legal_moves(self.current_player)
        if not next_moves:
            self.game_over = True
            self.winner = self.get_opponent(self.current_player)
            self.win_reason = f"{PLAYER_NAMES[self.current_player]} has no legal moves! {PLAYER_NAMES[self.winner]} wins!"
