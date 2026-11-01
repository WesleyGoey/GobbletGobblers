from typing import List, Tuple, Optional, Dict, NamedTuple
import copy

PLAYER_BLUE = 1    # Human (Player 1)
PLAYER_ORANGE = 2  # AI (Player 2)

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
    """Represents a game piece with player ownership and size (1=Small, 2=Medium, 3=Large)."""
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

def generate_winning_lines(win_streak: int = 5) -> List[List[Tuple[int, int]]]:
    """Generate all line segments of length win_streak on a 5x5 board (12 lines for streak=5)."""
    lines = []
    # Horizontal lines
    for r in range(5):
        for c in range(5 - win_streak + 1):
            lines.append([(r, c + i) for i in range(win_streak)])

    # Vertical lines
    for c in range(5):
        for r in range(5 - win_streak + 1):
            lines.append([(r + i, c) for i in range(win_streak)])

    # Main diagonals (top-left to bottom-right)
    for r in range(5 - win_streak + 1):
        for c in range(5 - win_streak + 1):
            lines.append([(r + i, c + i) for i in range(win_streak)])

    # Anti-diagonals (top-right to bottom-left)
    for r in range(5 - win_streak + 1):
        for c in range(win_streak - 1, 5):
            lines.append([(r + i, c - i) for i in range(win_streak)])

    return lines

WINNING_LINES_5 = generate_winning_lines(5)

class GameState:
    """Full representation of a 5x5 Gobblet Gobblers game state."""

    def __init__(self, win_streak: int = 5):
        self.win_streak: int = win_streak

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
        self.winning_lines: List[List[Tuple[int, int]]] = generate_winning_lines(win_streak)

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
    
    def find_all_winning_lines(self) -> Dict[int, List[List[Tuple[int, int]]]]:
        """Finds all winning lines on the board for each player according to self.win_streak."""
        wins: Dict[int, List[List[Tuple[int, int]]]] = {PLAYER_BLUE: [], PLAYER_ORANGE: []}
        for line in self.winning_lines:
            first_piece = self.get_top_piece(line[0][0], line[0][1])
            if not first_piece:
                continue
            p = first_piece.player
            match = True
            for r, c in line[1:]:
                piece = self.get_top_piece(r, c)
                if not piece or piece.player != p:
                    match = False
                    break
            if match:
                wins[p].append(line)
        return wins

    def check_winner_on_board(self) -> Tuple[Optional[int], Optional[List[Tuple[int, int]]]]:
        """Check visible pieces on board for winning lines."""
        wins = self.find_all_winning_lines()
        blue_wins = wins[PLAYER_BLUE]
        orange_wins = wins[PLAYER_ORANGE]

        if blue_wins and not orange_wins:
            return PLAYER_BLUE, blue_wins[0]
        if orange_wins and not blue_wins:
            return PLAYER_ORANGE, orange_wins[0]
        if blue_wins and orange_wins:
            opp = PLAYER_ORANGE if self.current_player == PLAYER_BLUE else PLAYER_BLUE
            return opp, wins[opp][0]
        return None, None

    def apply_move(self, move: Move) -> bool:
        """Execute a move on the game state. Updates board, reserves, win detection, and turn count."""
        if self.game_over or move.player != self.current_player:
            return False

        player = move.player
        opponent = PLAYER_ORANGE if player == PLAYER_BLUE else PLAYER_BLUE

        if move.move_type == 'drop':
            size = move.size
            if self.reserves[player][size] <= 0:
                return False
            r, c = move.to_pos
            if self.get_top_size(r, c) >= size:
                return False

            self.reserves[player][size] -= 1
            piece = Piece(player, size)
            if self.board[r][c]:
                move.covered_piece = self.board[r][c][-1]
                if self.covered_turns[r][c]:
                    self.covered_turns[r][c][-1] = self.turn_number
            self.board[r][c].append(piece)
            self.covered_turns[r][c].append(None)

        elif move.move_type == 'move':
            r1, c1 = move.from_pos
            r2, c2 = move.to_pos
            if (r1, c1) == (r2, c2):
                return False
            top1 = self.get_top_piece(r1, c1)
            if not top1 or top1.player != player or self.get_top_size(r2, c2) >= top1.size:
                return False

            piece = self.board[r1][c1].pop()
            self.covered_turns[r1][c1].pop()
            if self.board[r1][c1]:
                move.uncovered_piece = self.board[r1][c1][-1]
                self.covered_turns[r1][c1][-1] = None

            if self.board[r2][c2]:
                move.covered_piece = self.board[r2][c2][-1]
                if self.covered_turns[r2][c2]:
                    self.covered_turns[r2][c2][-1] = self.turn_number
            self.board[r2][c2].append(piece)
            self.covered_turns[r2][c2].append(None)

        # Cek kondisi menang
        winner, win_line = self.check_winner_on_board()
        if winner:
            self.winner = winner
            self.game_over = True
            self.winning_line = win_line
            self.win_reason = f"{PLAYER_NAMES[winner]} forms {self.win_streak}-in-a-row!"

        self.move_history.append((self.turn_number, move, str(move)))
        self.current_player = opponent
        self.turn_number += 1
        return True

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
        if not cell:
            return False, []

        piece = cell.pop()
        opponent = self.get_opponent(piece.player)

        # Check all winning lines that contain (r, c)
        opp_lines = []
        for line in self.winning_lines:
            if (r, c) in line:
                colors = []
                for lr, lc in line:
                    top = self.get_top_piece(lr, lc)
                    if top and top.player == opponent:
                        colors.append(top.player)
                    else:
                        break
                if len(colors) == self.win_streak:
                    opp_lines.append(line)

        cell.append(piece)  # Restore piece
        return (len(opp_lines) > 0), opp_lines

    def clone(self) -> 'GameState':
        """Deep copy of the game state for search simulations."""
        new_state = GameState(win_streak=self.win_streak)
        new_state.board = [[list(cell) for cell in row] for row in self.board]
        new_state.covered_turns = [[list(cell) for cell in row] for row in self.covered_turns]
        new_state.reserves = {
            p: dict(res) for p, res in self.reserves.items()
        }
        new_state.current_player = self.current_player
        new_state.turn_number = self.turn_number
        new_state.winner = self.winner
        new_state.game_over = self.game_over
        new_state.win_reason = self.win_reason
        new_state.winning_line = list(self.winning_line) if self.winning_line else None
        new_state.move_history = list(self.move_history)
        return new_state

    def get_legal_moves(self, player: Optional[int] = None) -> List[Move]:
        """Returns all legal moves including drops and board repositioning."""
        if self.game_over:
            return []
        if player is None:
            player = self.current_player

        legal_moves: List[Move] = []
        # 1. Action A: Reserve drops
        reserve = self.reserves[player]
        for size in [3, 2, 1]:
            if reserve[size] > 0:
                for r in range(5):
                    for c in range(5):
                        if self.get_top_size(r, c) < size:
                            legal_moves.append(Move(player=player, move_type='drop', size=size, to_pos=(r, c)))

        # 2. Action B: Move piece on board
        for r1 in range(5):
            for c1 in range(5):
                top_piece = self.get_top_piece(r1, c1)
                if top_piece and top_piece.player == player:
                    size = top_piece.size
                    for r2 in range(5):
                        for c2 in range(5):
                            if (r1, c1) != (r2, c2) and self.get_top_size(r2, c2) < size:
                                legal_moves.append(Move(player=player, move_type='move', size=size, from_pos=(r1, c1), to_pos=(r2, c2)))

        return legal_moves
