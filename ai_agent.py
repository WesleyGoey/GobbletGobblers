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
