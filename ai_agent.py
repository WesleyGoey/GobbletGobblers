import time
from typing import Optional, Tuple, Dict, Any, List
from game_engine import GameState, Move, PLAYER_BLUE, PLAYER_ORANGE

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
        self.nodes_evaluated = 0
        self.pruned_branches = 0
        self.candidate_moves.clear()

def get_best_move(self, real_state: GameState) -> Optional[Move]:
        """Find and return the optimal move from the current game state."""
        self.reset_telemetry()
        start_time = time.time()

        # Build belief state based on bounded memory
        state = build_belief_state(real_state, self.memory_window)
        legal_moves = state.get_legal_moves(self.player)
        if not legal_moves:
            return None

        best_score = -float('inf')
        best_move = legal_moves[0]

        for move in legal_moves:
            child = state.clone()
            child.apply_move(move)
            score = self._minimax(child, self.max_depth - 1, False)
            self.candidate_moves.append({'move': move, 'score': score})
            if score > best_score:
                best_score = score
                best_move = move

        self.last_search_time = time.time() - start_time
        return best_move
