"""
ai_agent.py - Minimax with Alpha-Beta Pruning, Bounded Rationality (Forgetting Memory),
and Real-time Live Decision Logging for 5x5 Gobblet Gobblers.

Key features:
1. Memory Window of 5 moves:
   - AI tracks the history of moves.
   - Buried pieces covered within the last 5 moves are remembered.
   - Buried pieces covered >5 moves ago are FORGOTTEN (treated as empty/unknown underneath in belief state).
2. Minimax + Alpha-Beta Pruning:
   - Operates on the AI's internal belief state.
   - Evaluates line potential (4-of-5, 3-of-5, 2-of-5), center control, reserve dominance, and trap risks.
3. Live Decision Telemetry:
   - Emits structured events for every node visit, evaluation, alpha-beta cutoff, and candidate move.
   - Enables real-time visualization of search trees, pruned branches, and memory belief.
"""

from typing import List, Tuple, Optional, Dict, Any, Callable
import random
import time
import math
import threading
from game_engine import (
    GameState, Piece, Move, WINNING_LINES, WINNING_LINES_5,
    CELL_LINES, CELL_LINES_5,
    PLAYER_BLUE, PLAYER_ORANGE, SIZE_SYMBOLS, SIZE_NAMES, PLAYER_NAMES
)


class RandomPlayer:
    """
    Simulated baseline opponent ('Pemain Acak') that selects legal moves uniformly at random.
    Used for baseline benchmarking, simulation matches, and testing.
    """

    def __init__(self, player: int = PLAYER_BLUE, seed: Optional[int] = None):
        """
        Initializes the random baseline agent with player color and random seed.

        Parameters:
            player: The player color assignment (PLAYER_BLUE or PLAYER_ORANGE).
            seed: Optional integer seed for deterministic pseudo-random action choices.
        """
        self.player = player
        self.rng = random.Random(seed)

    def get_action(
        self,
        state: GameState,
        step_callback: Optional[Callable[[str, Any], None]] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> Optional[Move]:
        """
        Selects a legal move uniformly at random from available actions.

        Acts as the benchmark opponent ('Pemain Acak') without heuristic evaluation
        or game tree search. Respects cancellation signals if provided.

        Parameters:
            state: The current GameState of the match.
            step_callback: Optional callback for telemetry events (unused by random agent).
            cancel_event: Optional threading.Event to abort search early if flagged.

        Returns:
            A randomly selected Move object, or None if no legal moves exist.
        """
        legal_moves = state.get_legal_moves(self.player)
        return self.rng.choice(legal_moves) if legal_moves else None


# Backward-compatibility alias
RandomAgent = RandomPlayer


class MemoryInspector:
    """Tracks and reports what the AI remembers vs what it has forgotten."""

    @staticmethod
    def get_memory_snapshot(game_state: GameState, memory_window: int = 5) -> List[Dict[str, Any]]:
        """
        Inspects all board cells with buried pieces to evaluate cognitive memory retention.

        Iterates over cells with stack depth > 1 and compares the turn each underlying piece
        was covered against the current turn. Pieces covered within `memory_window` turns
        are flagged as remembered, while older buried pieces are classified as forgotten.

        Parameters:
            game_state: The current GameState being evaluated.
            memory_window: The temporal threshold (in turns) for retaining memory of hidden pieces.

        Returns:
            A list of status dictionaries containing position, stack depth, piece references,
            temporal age, remembered boolean flag, and formatted display strings for UI telemetry.
        """
        current_turn = game_state.turn_number
        memory_records = []

        for r in range(5):
            for c in range(5):
                stack = game_state.get_cell_stack(r, c)
                if len(stack) > 1:
                    top_piece = stack[-1]
                    covered_turns = game_state.get_cell_covered_turns(r, c)
                    for idx, buried_piece in enumerate(stack[:-1]):
                        turn_covered = covered_turns[idx] if idx < len(covered_turns) else None
                        turns_ago = (current_turn - turn_covered) if turn_covered is not None else 999
                        is_remembered = (turns_ago <= memory_window)

                        p_col = "Blue" if buried_piece.player == PLAYER_BLUE else "Orange"
                        s_name = SIZE_NAMES.get(buried_piece.size, str(buried_piece.size)).split()[0]
                        rem_label = f"{s_name} {p_col}"

                        status_text = (
                            f"{rem_label} (remembered, {turns_ago}t ago)"
                            if is_remembered else
                            f"UNKNOWN (Forgotten >{memory_window}t)"
                        )

                        memory_records.append({
                            'pos': (r, c),
                            'depth': idx,
                            'top_piece': top_piece,
                            'buried_piece': buried_piece,
                            'turn_covered': turn_covered,
                            'turns_ago': turns_ago,
                            'is_remembered': is_remembered,
                            'status_text': status_text,
                            'display_str': (
                                f"Cell ({r},{c}) buried: {rem_label} (remembered)"
                                if is_remembered else
                                f"Cell ({r},{c}) buried piece: UNKNOWN (forgotten)"
                            )
                        })

        return memory_records


def build_belief_state(real_state: GameState, memory_window: int = 5) -> GameState:
    """
    Constructs the AI's internal partially-observable belief state based on bounded memory.

    Implements bounded rationality modeling:
    - Visible top pieces across all cells are fully perceived.
    - Buried pieces covered within the last `memory_window` turns are retained in memory
      and included in the belief stack.
    - Buried pieces covered more than `memory_window` turns ago are forgotten and omitted
      from the belief state (treated as empty space underneath).

    Parameters:
        real_state: The true ground-truth GameState.
        memory_window: The retention horizon (number of turns) for remembered covered pieces.

    Returns:
        A cloned GameState representing what the AI currently believes the board consists of.
    """
    belief = GameState(win_streak=real_state.win_streak)
    current_turn = real_state.turn_number

    for r in range(5):
        for c in range(5):
            real_stack = real_state.get_cell_stack(r, c)
            if not real_stack:
                belief.board[r][c] = []
                belief.covered_turns[r][c] = []
                continue

            top_piece = real_stack[-1]
            if len(real_stack) == 1:
                belief.board[r][c] = [top_piece]
                belief.covered_turns[r][c] = [None]
            else:
                covered_turns = real_state.get_cell_covered_turns(r, c)
                belief_stack = []
                belief_cov = []
                for idx, buried_piece in enumerate(real_stack[:-1]):
                    turn_covered = covered_turns[idx] if idx < len(covered_turns) else None
                    turns_ago = (current_turn - turn_covered) if turn_covered is not None else 999
                    if turns_ago <= memory_window:
                        # AI remembers this buried piece
                        belief_stack.append(buried_piece)
                        belief_cov.append(turn_covered)
                    else:
                        # AI has forgotten this buried piece! It is omitted from belief stack.
                        pass
                belief_stack.append(top_piece)
                belief_cov.append(None)
                belief.board[r][c] = belief_stack
                belief.covered_turns[r][c] = belief_cov

    # Copy reserves and turn metadata
    belief.reserves = {
        p: dict(res) for p, res in real_state.reserves.items()
    }
    belief.current_player = real_state.current_player
    belief.turn_number = real_state.turn_number
    belief.move_history = list(real_state.move_history)
    return belief


class AIHeuristic:
    """Evaluates a board state from the perspective of PLAYER_ORANGE (AI)."""

    CENTER_CELLS = {(2, 2)}
    INNER_RING = {(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)}

    @classmethod
    def evaluate(cls, state: GameState, ai_player: int = PLAYER_ORANGE) -> float:
        """
        Computes a comprehensive tactical static evaluation score for the board state.

        Heuristic Components:
        1. Terminal Win Detection:
           - Immediately returns +10000.0 if ai_player has won, or -10000.0 if opponent won.
        2. Line Potential & Gobble-Through Threats:
           - Analyzes all winning lines for uncontested streaks (4-of-5, 3-of-5, etc.).
           - Examines contested lines where an opponent's piece blocks a winning line but can be
             gobbled through by a larger piece in reserves or elsewhere on the board.
        3. Positional Center Control:
           - Grants bonuses for controlling the central cell (2, 2) and the surrounding inner ring.
        4. Reserve Dominance:
           - Weights unplayed reserve pieces (Large = 20, Medium = 10, Small = 3) reflecting
             their strategic value as responsive gobblers.
        5. Uncover Trap Risk & Hidden Liability:
           - Penalizes pinned friendly pieces that cover opponent pieces forming potential winning lines.
           - Rewards setting latent traps where opponent pieces cover friendly pieces.

        Parameters:
            state: The GameState (or belief state) to evaluate.
            ai_player: The player perspective to maximize (defaults to PLAYER_ORANGE).

        Returns:
            A floating-point evaluation score. Positive favors ai_player; negative favors opponent.
        """
        human_player = PLAYER_BLUE if ai_player == PLAYER_ORANGE else PLAYER_ORANGE
        board = state.board

        # 1. Pre-extract top pieces and top sizes for O(1) direct access
        top_grid = [[board[r][c][-1] if board[r][c] else None for c in range(5)] for r in range(5)]
        top_sizes = [[top_grid[r][c].size if top_grid[r][c] else 0 for c in range(5)] for r in range(5)]

        winning_lines = WINNING_LINES_5
        cell_lines_dict = CELL_LINES_5

        # Fast winner check on board
        blue_wins = []
        orange_wins = []
        for line in winning_lines:
            first = top_grid[line[0][0]][line[0][1]]
            if not first:
                continue
            p = first.player
            match = True
            for r, c in line[1:]:
                top = top_grid[r][c]
                if not top or top.player != p:
                    match = False
                    break
            if match:
                (blue_wins if p == PLAYER_BLUE else orange_wins).append(line)

        if blue_wins and not orange_wins:
            winner = PLAYER_BLUE
        elif orange_wins and not blue_wins:
            winner = PLAYER_ORANGE
        elif blue_wins and orange_wins:
            winner = state.get_opponent(state.current_player)
        else:
            winner = None

        if winner == ai_player:
            return 10000.0
        elif winner == human_player:
            return -10000.0

        score = 0.0
        human_res = state.reserves.get(human_player, {})
        ai_res = state.reserves.get(ai_player, {})
        max_human_res_size = max([s for s, cnt in human_res.items() if cnt > 0], default=0)
        max_ai_res_size = max([s for s, cnt in ai_res.items() if cnt > 0], default=0)

        # 1. Line Potential Analysis (across winning lines)
        for line in winning_lines:
            ai_count = 0
            human_count = 0
            empty_count = 0
            ai_pieces = []
            human_pieces = []

            for r, c in line:
                top = top_grid[r][c]
                if top is None:
                    empty_count += 1
                elif top.player == ai_player:
                    ai_count += 1
                    ai_pieces.append((r, c, top.size))
                else:
                    human_count += 1
                    human_pieces.append((r, c, top.size))

            # If line is uncontested by opponent:
            if human_count == 0 and ai_count > 0:
                if ai_count >= 4:
                    score += 650.0  # 4-of-5: Immediate winning threat on next turn!
                elif ai_count == 3:
                    score += 150.0  # 3-of-5: Strong threat & critical fork setup
                elif ai_count == 2:
                    score += 28.0   # 2-of-5
                elif ai_count >= 1:
                    score += 6.0    # 1-of-5

            elif ai_count == 0 and human_count > 0:
                if human_count >= 4:
                    score -= 820.0  # 4-of-5: Immediate lethal danger! Must block
                elif human_count == 3:
                    score -= 170.0  # 3-of-5: Severe human threat
                elif human_count == 2:
                    score -= 32.0   # 2-of-5
                elif human_count >= 1:
                    score -= 8.0    # 1-of-5

            # Contested lines: check for active gobble-through threats
            elif ai_count > 0 and human_count > 0:
                if human_count == 4 and ai_count == 1:
                    # 4 human pieces blocked by 1 AI piece: check if human can gobble it!
                    ai_pos_size = ai_pieces[0][2]
                    can_gobble = (max_human_res_size > ai_pos_size) or any(
                        top_sizes[br][bc] > ai_pos_size
                        for br in range(5) for bc in range(5)
                        if (br, bc) not in line and top_grid[br][bc] and top_grid[br][bc].player == human_player
                    )
                    if can_gobble:
                        score -= 780.0  # Human can gobble this piece and win immediately!
                    else:
                        score += 20.0   # Solid secure block by AI Large piece
                elif ai_count == 4 and human_count == 1:
                    # 4 AI pieces blocked by 1 human piece: check if AI can gobble it!
                    human_pos_size = human_pieces[0][2]
                    can_gobble = (max_ai_res_size > human_pos_size) or any(
                        top_sizes[br][bc] > human_pos_size
                        for br in range(5) for bc in range(5)
                        if (br, bc) not in line and top_grid[br][bc] and top_grid[br][bc].player == ai_player
                    )
                    if can_gobble:
                        score += 620.0  # AI can gobble and win on next turn!
                    else:
                        score -= 20.0
                elif human_count == 3 and ai_count == 1 and empty_count == 1:
                    # 3 human + 1 AI (gobbleable) + 1 empty
                    ai_pos_size = ai_pieces[0][2]
                    if max_human_res_size > ai_pos_size:
                        score -= 140.0  # Live 3-of-5 threat
                elif ai_count == 3 and human_count == 1 and empty_count == 1:
                    # 3 AI + 1 human (gobbleable) + 1 empty
                    human_pos_size = human_pieces[0][2]
                    if max_ai_res_size > human_pos_size:
                        score += 125.0  # Live 3-of-5 threat
                else:
                    diff = ai_count - human_count
                    score += diff * 5.0

        # 2. Board Positional Advantage (Center control)
        top_center = top_grid[2][2]
        if top_center:
            score += (35.0 if top_center.player == ai_player else -35.0)

        for r, c in cls.INNER_RING:
            top = top_grid[r][c]
            if top:
                score += (15.0 if top.player == ai_player else -15.0)

        # 3. Reserve Advantage
        # Large (Size 3) pieces are the most powerful gobblers
        ai_res = state.reserves.get(ai_player, {})
        human_res = state.reserves.get(human_player, {})

        res_val_ai = ai_res.get(3, 0) * 20.0 + ai_res.get(2, 0) * 10.0 + ai_res.get(1, 0) * 3.0
        res_val_human = human_res.get(3, 0) * 20.0 + human_res.get(2, 0) * 10.0 + human_res.get(1, 0) * 3.0
        score += (res_val_ai - res_val_human)

        # 4. Trap Risk & Uncover Vulnerability Analysis
        # Evaluates hidden uncover liabilities (pinned AI pieces) and latent traps (human covering AI)
        for r in range(5):
            for c in range(5):
                cell = board[r][c]
                if len(cell) > 1:
                    top_piece = cell[-1]
                    covered_piece = cell[-2]

                    # Case A: AI piece covers a human piece (Pinned AI piece / Uncover trap risk!)
                    if top_piece.player == ai_player and covered_piece.player == human_player:
                        for line in cell_lines_dict[(r, c)]:
                            human_in_line = 0
                            for lr, lc in line:
                                if (lr, lc) != (r, c):
                                    top = top_grid[lr][lc]
                                    if top and top.player == human_player:
                                        human_in_line += 1
                            if human_in_line >= 4:
                                score -= 750.0
                            elif human_in_line == 3:
                                score -= 180.0
                            elif human_in_line == 2:
                                score -= 50.0
                            elif human_in_line >= 1:
                                score -= 15.0

                    # Case B: Human piece covers an AI piece (Latent trap set for human!)
                    elif top_piece.player == human_player and covered_piece.player == ai_player:
                        for line in cell_lines_dict[(r, c)]:
                            ai_in_line = 0
                            for lr, lc in line:
                                if (lr, lc) != (r, c):
                                    top = top_grid[lr][lc]
                                    if top and top.player == ai_player:
                                        ai_in_line += 1
                            if ai_in_line >= 4:
                                score += 600.0
                            elif ai_in_line == 3:
                                score += 150.0
                            elif ai_in_line == 2:
                                score += 40.0
                            elif ai_in_line >= 1:
                                score += 10.0

                    # Case C: Friendly stack (AI piece covers AI piece)
                    elif top_piece.player == ai_player and covered_piece.player == ai_player:
                        score += 10.0

        return score


class SearchTelemetry:
    """Holds real-time search state and decision logs for the live GUI panel (thread-safe)."""

    def __init__(self):
        """
        Serves as the constructor and thread-safe state container initializer for SearchTelemetry.

        Establishes the mutual exclusion primitive (`threading.Lock`) used to synchronize shared
        search state between background Minimax computation threads and the main GUI rendering thread
        (60 FPS event loop). Immediately delegates to `reset()` to initialize all search counters,
        timing trackers, candidate score maps, pruned branch logs, and cognitive memory snapshots.
        """
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        """
        Resets all real-time search metrics and candidate move tables to their baseline states.

        Clears node visit counters, prune counters, timers, candidate move score maps,
        current search branch trails, and memory snapshots.
        """
        with self.lock:
            self.is_searching: bool = False
            self.current_depth: int = 0
            self.max_depth: int = 3
            self.nodes_visited: int = 0
            self.prune_count: int = 0
            self.start_time: float = 0.0
            self.elapsed_time: float = 0.0

            # Tree logs and candidate moves
            self.candidate_scores: Dict[str, float] = {}  # move_str -> score
            self.current_branch: List[str] = []           # moves in current search path
            self.tree_log: List[Dict[str, Any]] = []      # recent node visits / pruned events
            self.pruned_branches: List[str] = []          # descriptions of pruned cuts
            self.best_move: Optional[Move] = None
            self.best_score: float = -math.inf
            self.memory_snapshot: List[Dict[str, Any]] = []

    def get_candidate_scores_copy(self) -> List[Tuple[str, float]]:
        """
        Thread-safely acquires and returns a shallow copy of root candidate moves and scores.

        Returns:
            A list of (move_str, evaluation_score) tuples representing root actions explored so far.
        """
        with self.lock:
            return list(self.candidate_scores.items())

    def get_pruned_branches_copy(self) -> List[str]:
        """
        Thread-safely acquires and returns a shallow copy of logged Alpha-Beta pruning cutoff events.

        Returns:
            A list of formatted strings describing recent cutoff occurrences and threshold values.
        """
        with self.lock:
            return list(self.pruned_branches)

    def get_current_branch_copy(self) -> List[str]:
        """
        Thread-safely acquires and returns a shallow copy of the active search tree branch path.

        Returns:
            A list of move strings forming the sequence from root to the currently explored node.
        """
        with self.lock:
            return list(self.current_branch)

    def get_memory_snapshot_copy(self) -> List[Dict[str, Any]]:
        """
        Thread-safely acquires and returns a copy of the AI cognitive memory status records.

        Returns:
            A list of dictionaries describing buried pieces, temporal ages, and retention states.
        """
        with self.lock:
            return list(self.memory_snapshot)


def order_moves_heuristically(moves: List[Move]) -> None:
    """
    Sorts moves in-place using a lightweight tactical heuristic to optimize Alpha-Beta pruning.

    Orders moves by piece size (preferring larger drops/gobbles) and central board positioning
    (center cell (2, 2) and inner ring cells). Prioritizing high-utility moves increases the
    likelihood of early alpha-beta cutoffs, significantly pruning the game tree search space.

    Parameters:
        moves: The list of Move objects to be ordered in-place.
    """
    inner_ring = AIHeuristic.INNER_RING

    def move_score(m: Move) -> int:
        """
        Serves as the internal heuristic ranking function for Alpha-Beta move ordering.

        Computes an integer utility score to sort moves in descending order of tactical priority
        before expanding search tree branches. Prioritizes:
        1. Piece size dominance: `(m.size or 0) * 10` favors larger pieces that dominate/gobble cells.
        2. Strategic board centrality: +5 bonus for the pivotal center cell (2, 2) and +2 bonus for
           the surrounding inner ring cells.

        By evaluating the strongest moves first, Alpha-Beta pruning maximizes the frequency of beta
        cutoffs, substantially shrinking the branching factor and search latency.

        Parameters:
            m: The Move instance being scored.

        Returns:
            An integer heuristic score used as the sorting key.
        """
        sc = (m.size or 0) * 10
        to_pos = m.to_pos
        if to_pos == (2, 2):
            sc += 5
        elif to_pos in inner_ring:
            sc += 2
        return sc

    moves.sort(key=move_score, reverse=True)


class MinimaxAgent:
    """
    Minimax agent with Alpha-Beta Pruning, Bounded Rationality (5-move Memory Window),
    and live real-time search event callbacks.
    """

    def __init__(
        self,
        player: int = PLAYER_ORANGE,
        max_depth: int = 3,
        memory_window: int = 5,
        telemetry: Optional[SearchTelemetry] = None,
        enable_pruning: bool = True
    ):
        """
        Initializes the Minimax AI Agent with Alpha-Beta pruning and bounded rationality.

        Parameters:
            player: Player assignment (PLAYER_ORANGE by default).
            max_depth: Maximum ply depth for minimax game tree exploration (e.g. 1, 2, or 3).
            memory_window: Number of turns hidden pieces remain remembered in the belief state.
            telemetry: Shared SearchTelemetry instance for live GUI metrics streaming.
            enable_pruning: Flag enabling Alpha-Beta branch cutoffs (True for standard play;
                            can be disabled to benchmark against Pure Minimax).
        """
        self.player = player
        self.max_depth = max_depth
        self.memory_window = memory_window
        self.telemetry = telemetry if telemetry is not None else SearchTelemetry()
        self.enable_pruning = enable_pruning
        self.step_delay: float = 0.005  # Slight throttle for visually satisfying live updates

    def get_action(
        self,
        real_state: GameState,
        step_callback: Optional[Callable[[str, Any], None]] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> Optional[Move]:
        """
        Executes the root Minimax search on the AI's internal belief state to select the best move.

        Process:
        1. Captures memory snapshot and constructs the bounded rationality belief state.
        2. Gathers and heuristically orders all legal moves available to the AI.
        3. Evaluates each candidate move by simulating the action and exploring the game tree
           via `_min_value`, updating alpha bounds.
        4. Dispatches live telemetry events via `step_callback` for UI visualization.
        5. Supports graceful search cancellation if `cancel_event` is signaled.

        Parameters:
            real_state: The actual ground-truth GameState.
            step_callback: Optional callable for streaming real-time search events to GUI.
            cancel_event: Optional threading.Event to abort calculation prematurely.

        Returns:
            The optimal Move selected by Minimax, or None if no legal moves exist.
        """
        search_depth = self.max_depth
        search_memory = self.memory_window

        self.telemetry.reset()
        self.telemetry.is_searching = True
        self.telemetry.max_depth = search_depth
        self.telemetry.start_time = time.time()

        # Step 1: Capture and log memory state
        memory_records = MemoryInspector.get_memory_snapshot(real_state, search_memory)
        self.telemetry.memory_snapshot = memory_records
        if step_callback:
            step_callback('MEMORY_STATE', memory_records)

        # Step 2: Build internal belief state with forgetting memory
        belief_state = build_belief_state(real_state, search_memory)

        # Step 3: Get legal moves
        legal_moves = belief_state.get_legal_moves(self.player)
        if not legal_moves:
            self.telemetry.is_searching = False
            return None

        # Sort moves heuristically (drops of size 3 first, then center moves)
        order_moves_heuristically(legal_moves)

        best_move = legal_moves[0]
        alpha = -math.inf
        beta = math.inf
        best_score = -math.inf

        # Root level search
        for move in legal_moves:
            if cancel_event and cancel_event.is_set():
                with self.telemetry.lock:
                    self.telemetry.is_searching = False
                return None

            move_str = str(move)
            with self.telemetry.lock:
                self.telemetry.current_branch = [move_str]
                self.telemetry.nodes_visited += 1

            if step_callback:
                step_callback('EXPLORING_ROOT', {
                    'move': move,
                    'move_str': move_str,
                    'alpha': alpha,
                    'beta': beta
                })

            # Simulate move on cloned belief state
            next_state = belief_state.clone()
            next_state.apply_move(move)

            # Check if this move wins immediately
            if next_state.winner == self.player:
                score = 10000.0
            else:
                score = self._min_value(
                    next_state,
                    depth=1,
                    alpha=alpha,
                    beta=beta,
                    path=[move_str],
                    step_callback=step_callback,
                    search_depth=search_depth,
                    cancel_event=cancel_event
                )

            if cancel_event and cancel_event.is_set():
                with self.telemetry.lock:
                    self.telemetry.is_searching = False
                return None

            with self.telemetry.lock:
                self.telemetry.candidate_scores[move_str] = round(score, 1)
                if score > best_score:
                    best_score = score
                    best_move = move
                    self.telemetry.best_move = best_move
                    self.telemetry.best_score = round(best_score, 1)

            alpha = max(alpha, best_score)

            if step_callback:
                step_callback('ROOT_EVAL', {
                    'move': move,
                    'move_str': move_str,
                    'score': score,
                    'best_move': best_move,
                    'best_score': best_score
                })

            if self.step_delay > 0:
                time.sleep(self.step_delay)

        with self.telemetry.lock:
            self.telemetry.is_searching = False
            self.telemetry.elapsed_time = time.time() - self.telemetry.start_time
            self.telemetry.best_move = best_move
            self.telemetry.best_score = round(best_score, 1)

        if step_callback:
            step_callback('SEARCH_COMPLETE', {
                'best_move': best_move,
                'best_score': best_score,
                'nodes_visited': self.telemetry.nodes_visited,
                'prune_count': self.telemetry.prune_count,
                'elapsed_time': self.telemetry.elapsed_time
            })

        return best_move

    def _max_value(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        path: List[str],
        step_callback: Optional[Callable[[str, Any], None]],
        search_depth: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> float:
        """
        Executes the maximizing step of the Minimax algorithm (AI's turn).

        Explores all legal moves for `self.player`, recursively calling `_min_value`
        to evaluate opponent responses. Applies Alpha-Beta pruning when the current evaluation
        exceeds or equals beta.

        Parameters:
            state: The current GameState node in the search tree.
            depth: Current ply depth in the tree.
            alpha: Current best lower bound achieved by the maximizer.
            beta: Current best upper bound achieved by the minimizer.
            path: Move sequence history leading to this node.
            step_callback: Telemetry streaming callback.
            search_depth: Maximum depth limit for the active search.
            cancel_event: Abort signal flag.

        Returns:
            The maximum evaluated heuristic score for this node.
        """
        if cancel_event and cancel_event.is_set():
            return 0.0

        target_depth = search_depth if search_depth is not None else self.max_depth

        with self.telemetry.lock:
            self.telemetry.nodes_visited += 1

        if state.game_over:
            if state.winner == self.player:
                return 10000.0 - depth * 10
            elif state.winner is not None:
                return -10000.0 + depth * 10
            return 0.0

        if depth >= target_depth:
            return AIHeuristic.evaluate(state, self.player)

        moves = state.get_legal_moves(self.player)
        if not moves:
            return AIHeuristic.evaluate(state, self.player)

        # Move ordering heuristic
        order_moves_heuristically(moves)

        value = -math.inf
        for move in moves:
            if cancel_event and cancel_event.is_set():
                return 0.0

            move_str = str(move)
            new_path = path + [move_str]
            with self.telemetry.lock:
                self.telemetry.current_branch = new_path

            if step_callback and self.telemetry.nodes_visited % 15 == 0:
                step_callback('NODE_VISIT', {
                    'depth': depth,
                    'path': new_path,
                    'alpha': alpha,
                    'beta': beta
                })

            next_state = state.clone()
            next_state.apply_move(move)

            v = self._min_value(
                next_state, depth + 1, alpha, beta, new_path, step_callback,
                search_depth=target_depth, cancel_event=cancel_event
            )
            value = max(value, v)

            if self.enable_pruning and value >= beta:
                prune_msg = f"[PRUNED] at d={depth}: {move_str} (score {value:.1f} >= beta {beta:.1f})"
                with self.telemetry.lock:
                    self.telemetry.prune_count += 1
                    self.telemetry.pruned_branches.append(prune_msg)
                    if len(self.telemetry.pruned_branches) > 8:
                        self.telemetry.pruned_branches.pop(0)

                if step_callback:
                    step_callback('PRUNED', {
                        'depth': depth,
                        'move_str': move_str,
                        'message': prune_msg
                    })
                return value

            alpha = max(alpha, value)

        return value

    def _min_value(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        path: List[str],
        step_callback: Optional[Callable[[str, Any], None]],
        search_depth: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> float:
        """
        Executes the minimizing step of the Minimax algorithm (Human opponent's turn).

        Explores all legal moves for the opponent, recursively calling `_max_value`
        to evaluate AI counter-moves. Applies Alpha-Beta pruning when the current evaluation
        drops below or equals alpha.

        Parameters:
            state: The current GameState node in the search tree.
            depth: Current ply depth in the tree.
            alpha: Current best lower bound achieved by the maximizer.
            beta: Current best upper bound achieved by the minimizer.
            path: Move sequence history leading to this node.
            step_callback: Telemetry streaming callback.
            search_depth: Maximum depth limit for the active search.
            cancel_event: Abort signal flag.

        Returns:
            The minimum evaluated heuristic score for this node.
        """
        if cancel_event and cancel_event.is_set():
            return 0.0

        target_depth = search_depth if search_depth is not None else self.max_depth

        with self.telemetry.lock:
            self.telemetry.nodes_visited += 1

        human_player = PLAYER_BLUE if self.player == PLAYER_ORANGE else PLAYER_ORANGE

        if state.game_over:
            if state.winner == self.player:
                return 10000.0 - depth * 10
            elif state.winner is not None:
                return -10000.0 + depth * 10
            return 0.0

        if depth >= target_depth:
            return AIHeuristic.evaluate(state, self.player)

        moves = state.get_legal_moves(human_player)
        if not moves:
            return AIHeuristic.evaluate(state, self.player)

        # Move ordering heuristic
        order_moves_heuristically(moves)

        value = math.inf
        for move in moves:
            if cancel_event and cancel_event.is_set():
                return 0.0

            move_str = str(move)
            new_path = path + [move_str]
            with self.telemetry.lock:
                self.telemetry.current_branch = new_path

            if step_callback and self.telemetry.nodes_visited % 15 == 0:
                step_callback('NODE_VISIT', {
                    'depth': depth,
                    'path': new_path,
                    'alpha': alpha,
                    'beta': beta
                })

            next_state = state.clone()
            next_state.apply_move(move)

            v = self._max_value(
                next_state, depth + 1, alpha, beta, new_path, step_callback,
                search_depth=target_depth, cancel_event=cancel_event
            )
            value = min(value, v)

            if self.enable_pruning and value <= alpha:
                prune_msg = f"[PRUNED] at d={depth}: {move_str} (score {value:.1f} <= alpha {alpha:.1f})"
                with self.telemetry.lock:
                    self.telemetry.prune_count += 1
                    self.telemetry.pruned_branches.append(prune_msg)
                    if len(self.telemetry.pruned_branches) > 8:
                        self.telemetry.pruned_branches.pop(0)

                if step_callback:
                    step_callback('PRUNED', {
                        'depth': depth,
                        'move_str': move_str,
                        'message': prune_msg
                    })
                return value

            beta = min(beta, value)

        return value
