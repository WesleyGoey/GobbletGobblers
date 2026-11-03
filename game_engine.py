"""
game_engine.py - Core game state, rules, move validation, and win checking
for 5x5 Gobblet Gobblers.

Specifications:
- 5x5 board (25 cells)
- Each cell is a LIFO stack (up to 3 nested pieces, strictly descending size: largest on top)
- 2 Players: Player 1 (Blue / Human), Player 2 (Orange / AI)
- 12 pieces per player: 4 Large (size 3), 4 Medium (size 2), 4 Small (size 1)
- Win condition: 5-in-a-row (default) or 4-in-a-row of visible pieces
- Touch-move & Uncover Trap rule support
"""

from typing import List, Tuple, Optional, Dict, NamedTuple
import copy

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
        self.player = player
        self.move_type = move_type
        self.to_pos = to_pos
        self.size = size
        self.from_pos = from_pos
        # Captured metadata for undo / history tracking
        self.uncovered_piece: Optional[Piece] = None
        self.covered_piece: Optional[Piece] = None

    def __str__(self) -> str:
        r2, c2 = self.to_pos
        p_name = "Blue" if self.player == PLAYER_BLUE else "Orange"
        if self.move_type == 'drop':
            return f"[{p_name}] Drop {SIZE_SYMBOLS[self.size]} -> ({r2},{c2})"
        else:
            r1, c1 = self.from_pos
            s_str = f"S{self.size}" if self.size else "Piece"
            return f"[{p_name}] Move {s_str} ({r1},{c1}) -> ({r2},{c2})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
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
        return hash((self.player, self.move_type, self.to_pos, self.size, self.from_pos))


def generate_winning_lines(win_streak: int = 4) -> List[List[Tuple[int, int]]]:
    """
    Generate all line segments of length win_streak on a 5x5 board.
    - win_streak=4: 28 lines (10 horizontal + 10 vertical + 8 diagonal).
    - win_streak=5: 12 lines (5 horizontal + 5 vertical + 2 diagonal).
    """
    if win_streak < 3 or win_streak > 5:
        raise ValueError("win_streak must be between 3 and 5 for a 5x5 board")

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


# Precomputed cached line configurations for standard 5x5 board
WINNING_LINES_4 = generate_winning_lines(4)
WINNING_LINES_5 = generate_winning_lines(5)

# Kept for backwards compatibility: default 4-in-a-row lines (28 lines)
WINNING_LINES = WINNING_LINES_4

# Precomputed cell-to-lines index for O(1) lookup of lines intersecting any cell (r, c)
CELL_LINES_4: Dict[Tuple[int, int], List[List[Tuple[int, int]]]] = {
    (r, c): [line for line in WINNING_LINES_4 if (r, c) in line]
    for r in range(5) for c in range(5)
}
CELL_LINES_5: Dict[Tuple[int, int], List[List[Tuple[int, int]]]] = {
    (r, c): [line for line in WINNING_LINES_5 if (r, c) in line]
    for r in range(5) for c in range(5)
}


class GameState:
    """Full representation of a 5x5 Gobblet Gobblers game state."""

    def __init__(self, win_streak: int = 5):
        self.win_streak: int = win_streak
        if win_streak == 4:
            self.winning_lines = WINNING_LINES_4
            self.cell_lines = CELL_LINES_4
        elif win_streak == 5:
            self.winning_lines = WINNING_LINES_5
            self.cell_lines = CELL_LINES_5
        else:
            self.winning_lines = generate_winning_lines(win_streak)
            self.cell_lines = {
                (r, c): [line for line in self.winning_lines if (r, c) in line]
                for r in range(5) for c in range(5)
            }

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
        cell = self.board[r][c]
        return cell[-1] if cell else None

    def get_top_size(self, r: int, c: int) -> int:
        piece = self.get_top_piece(r, c)
        return piece.size if piece else 0

    def get_stack_depth(self, r: int, c: int) -> int:
        return len(self.board[r][c])

    def get_cell_stack(self, r: int, c: int) -> List[Piece]:
        return list(self.board[r][c])

    def get_cell_covered_turns(self, r: int, c: int) -> List[Optional[int]]:
        return list(self.covered_turns[r][c])

    def set_cell(
        self,
        r: int,
        c: int,
        pieces: List[Piece],
        covered_turns: Optional[List[Optional[int]]] = None
    ):
        """Helper for initializing or setting up specific cell stacks in tests."""
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
        """Deep copy of the game state for fast search simulations."""
        new_state = GameState.__new__(GameState)
        new_state.win_streak = self.win_streak
        new_state.winning_lines = self.winning_lines
        new_state.cell_lines = getattr(self, 'cell_lines', None)
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
        Finds all winning lines on the board for each player according to self.win_streak.
        Returns a dict mapping player -> list of winning lines.
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
        Check visible pieces on board for winning lines.
        Returns (winner, winning_line) or (None, None).
        If both have lines, resolves according to rules giving priority to opponent of active player.
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
        return PLAYER_ORANGE if player == PLAYER_BLUE else PLAYER_BLUE

    def would_lift_uncover_opponent_win(self, from_pos: Tuple[int, int]) -> Tuple[bool, List[List[Tuple[int, int]]]]:
        """
        Simulate lifting the top piece at from_pos.
        Checks if doing so reveals any opponent winning lines.
        Returns (is_uncover_trap, list_of_opponent_winning_lines).
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
        Get all legal moves for the given player (defaults to current_player).
        Accounts for:
        1. Reserve drops onto empty cells or smaller pieces.
        2. Board piece relocations onto empty cells or smaller pieces.
        3. The Uncover Trap rule: If lifting a piece reveals an opponent's 4-in-a-row,
           the move MUST block/break all revealed lines.
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
        Execute a move on the game state.
        Updates board, reserves, win detection, uncover traps, and turn count.
        Returns True if successful, False if illegal.
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
        """Record move in history, switch active player, and check legal moves."""
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
