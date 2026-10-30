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
