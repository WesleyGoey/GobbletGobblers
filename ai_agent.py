import time
from typing import Optional, Tuple, Dict, Any, List
from game_engine import GameState, Move, PLAYER_BLUE, PLAYER_ORANGE


def build_belief_state(real_state: GameState, memory_window: int = 5) -> GameState:
    """
    Constructs the AI's internal belief state based on its bounded memory.
    - Top pieces on the board are always visible.
    - Buried pieces covered within the last memory_window moves are remembered.
    - Buried pieces covered > memory_window moves ago are forgotten.
    """
    belief = real_state.clone()
    current_turn = real_state.turn_number

    for r in range(5):
        for c in range(5):
            real_stack = real_state.get_cell_stack(r, c)
            if len(real_stack) > 1:
                covered_turns = real_state.get_cell_covered_turns(r, c)
                filtered_stack = []
                filtered_turns = []
                for idx, piece in enumerate(real_stack[:-1]):
                    t_cov = covered_turns[idx] if idx < len(covered_turns) else None
                    if t_cov is not None and (current_turn - t_cov) <= memory_window:
                        filtered_stack.append(piece)
                        filtered_turns.append(t_cov)
                filtered_stack.append(real_stack[-1])
                filtered_turns.append(None)
                belief.board[r][c] = filtered_stack
                belief.covered_turns[r][c] = filtered_turns

    return belief


class MinimaxAgent:
    """Adversarial AI agent using Minimax search and bounded rationality."""

    def __init__(self, player: int = PLAYER_ORANGE, max_depth: int = 3, memory_window: int = 5):
        self.player = player
        self.max_depth = max_depth
        self.memory_window = memory_window
        self.nodes_evaluated = 0
        self.pruned_branches = 0
        self.last_search_time = 0.0
        self.candidate_moves: List[Dict[str, Any]] = []

    def reset_telemetry(self):
        """Reset search performance counters before each search execution."""
        self.nodes_evaluated = 0
        self.pruned_branches = 0
        self.candidate_moves.clear()

    def evaluate_state(self, state: GameState) -> float:
        """Basic heuristic evaluation of board state from the perspective of self.player."""
        winner, _ = state.check_winner_on_board()
        if winner == self.player:
            return 10000.0
        elif winner is not None:
            return -10000.0

        score = 0.0
        for r in range(5):
            for c in range(5):
                top = state.get_top_piece(r, c)
                if top:
                    val = top.size * 2
                    score += val if top.player == self.player else -val
        return score

    def _minimax(self, state: GameState, depth: int, is_maximizing: bool) -> float:
        """Recursive adversarial minimax traversal over game states."""
        self.nodes_evaluated += 1

        winner, _ = state.check_winner_on_board()
        if winner == self.player:
            return 10000.0 - (self.max_depth - depth)
        elif winner is not None:
            return -10000.0 + (self.max_depth - depth)
        if depth == 0 or state.game_over:
            return self.evaluate_state(state)

        legal_moves = state.get_legal_moves()
        if not legal_moves:
            return 0.0

        if is_maximizing:
            max_eval = -float('inf')
            for move in legal_moves:
                child = state.clone()
                child.apply_move(move)
                ev = self._minimax(child, depth - 1, False)
                max_eval = max(max_eval, ev)
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                child = state.clone()
                child.apply_move(move)
                ev = self._minimax(child, depth - 1, True)
                min_eval = min(min_eval, ev)
            return min_eval

    def order_moves(self, state: GameState, moves: List[Move]) -> List[Move]:
        """Order moves heuristically so best branches are evaluated first, maximizing cutoffs."""
        def move_weight(m: Move) -> int:
            weight = 0
            r, c = m.to_pos
            # Center control priority
            if (r, c) == (2, 2):
                weight += 50
            elif (r, c) in {(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)}:
                weight += 20
            # Gobbling opponent piece priority
            top = state.get_top_piece(r, c)
            if top and top.player != self.player:
                weight += 40 * top.size
            # Size priority
            if m.size:
                weight += m.size * 5
            return weight

        return sorted(moves, key=move_weight, reverse=True)

    def _alpha_beta(self, state: GameState, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        """Minimax with Alpha-Beta Pruning to skip suboptimal search branches."""
        self.nodes_evaluated += 1

        winner, _ = state.check_winner_on_board()
        if winner == self.player:
            return 10000.0 - (self.max_depth - depth)
        elif winner is not None:
            return -10000.0 + (self.max_depth - depth)
        if depth == 0 or state.game_over:
            return self.evaluate_state(state)

        legal_moves = self.order_moves(state, state.get_legal_moves())
        if not legal_moves:
            return 0.0

        if is_maximizing:
            max_eval = -float('inf')
            for move in legal_moves:
                child = state.clone()
                child.apply_move(move)
                ev = self._alpha_beta(child, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, ev)
                alpha = max(alpha, ev)
                if beta <= alpha:
                    self.pruned_branches += 1
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                child = state.clone()
                child.apply_move(move)
                ev = self._alpha_beta(child, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, ev)
                beta = min(beta, ev)
                if beta <= alpha:
                    self.pruned_branches += 1
                    break
            return min_eval

    def get_best_move(self, real_state: GameState) -> Optional[Move]:
        """Find and return the optimal move from the current game state."""
        self.reset_telemetry()
        start_time = time.time()

        # Build belief state based on bounded memory
        state = build_belief_state(real_state, self.memory_window)
        legal_moves = self.order_moves(state, state.get_legal_moves(self.player))
        if not legal_moves:
            return None

        best_score = -float('inf')
        best_move = legal_moves[0]
        alpha = -float('inf')
        beta = float('inf')

        for move in legal_moves:
            child = state.clone()
            child.apply_move(move)
            score = self._alpha_beta(child, self.max_depth - 1, alpha, beta, False)
            self.candidate_moves.append({'move': move, 'score': score})
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)

        self.last_search_time = time.time() - start_time
        return best_move

class AIHeuristic:
    """Evaluates a board state from the perspective of PLAYER_ORANGE (AI)."""

    @classmethod
    def evaluate(cls, state: GameState, ai_player: int = PLAYER_ORANGE) -> float:
        human_player = PLAYER_BLUE if ai_player == PLAYER_ORANGE else PLAYER_ORANGE
        winner, _ = state.check_winner_on_board()
        if winner == ai_player:
            return 10000.0
        elif winner == human_player:
            return -10000.0

        score = 0.0
        for line in state.winning_lines:
            ai_count = sum(1 for r, c in line if state.get_top_piece(r, c) and state.get_top_piece(r, c).player == ai_player)
            human_count = sum(1 for r, c in line if state.get_top_piece(r, c) and state.get_top_piece(r, c).player == human_player)

            if human_count == 0 and ai_count > 0:
                if ai_count >= 4: score += 650.0
                elif ai_count == 3: score += 150.0
                elif ai_count == 2: score += 28.0
            elif ai_count == 0 and human_count > 0:
                if human_count >= 4: score -= 820.0  # High defense priority
                elif human_count == 3: score -= 170.0
                elif human_count == 2: score -= 32.0

        # Center control bonus
        center = state.get_top_piece(2, 2)
        if center:
            score += 35.0 if center.player == ai_player else -35.0

        return score
