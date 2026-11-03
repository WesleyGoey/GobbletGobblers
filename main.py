"""
main.py - Entry point for 5x5 Gobblet Gobblers.

Usage:
  python main.py          # Launches full Pygame GUI with interactive board & live AI panel
  python main.py --cli    # Runs a quick CLI simulation game (AI vs AI or test)
"""

import sys
import argparse
from typing import Optional
from game_engine import GameState, PLAYER_BLUE, PLAYER_ORANGE, PLAYER_NAMES
from ai_agent import MinimaxAgent


DIFFICULTY_MAP = {
    'easy': {'depth': 1, 'memory_window': 3},
    'medium': {'depth': 2, 'memory_window': 5},
    'hard': {'depth': 3, 'memory_window': 8},
}


def run_cli_match(win_streak: int = 5, difficulty: str = "medium", depth: Optional[int] = None, max_turns: int = 40):
    diff = difficulty.lower() if isinstance(difficulty, str) and difficulty.lower() in DIFFICULTY_MAP else "medium"
    cfg = DIFFICULTY_MAP[diff]
    search_depth = depth if depth is not None else cfg['depth']
    mem_win = cfg['memory_window']

    print("=" * 60)
    print(f" Gobblet Gobblers 5x5 - CLI Simulation Match ({win_streak}-in-a-row)")
    print(f" Difficulty: {diff.upper()} (Depth {search_depth}, Memory Window {mem_win})")
    print(" Player 1 (Blue) vs Player 2 (Orange AI)")
    print("=" * 60)

    state = GameState(win_streak=win_streak)
    blue_agent = MinimaxAgent(player=PLAYER_BLUE, max_depth=search_depth, memory_window=mem_win)
    orange_agent = MinimaxAgent(player=PLAYER_ORANGE, max_depth=search_depth, memory_window=mem_win)
    blue_agent.step_delay = 0
    orange_agent.step_delay = 0

    while not state.game_over and state.turn_number <= max_turns:
        active_p = state.current_player
        agent = blue_agent if active_p == PLAYER_BLUE else orange_agent
        p_name = PLAYER_NAMES[active_p]

        print(f"\n--- Turn {state.turn_number}: {p_name} ---")
        move = agent.get_action(state)
        if not move:
            print(f"No legal moves for {p_name}! Match ends.")
            break

        print(f"Action: {move}")
        state.apply_move(move)

        # Print current visible board
        for r in range(5):
            row_str = " | ".join(str(state.get_top_piece(r, c) or "  .  ") for c in range(5))
            print(f"  [{row_str}]")

    print("\n" + "=" * 60)
    if state.game_over:
        print(f"GAME OVER! Winner: {PLAYER_NAMES[state.winner]}")
        print(f"Reason: {state.win_reason}")
    else:
        print("Match reached turn limit without a winner.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="5x5 Gobblet Gobblers with Minimax & Bounded Rationality")
    parser.add_argument("--cli", action="store_true", help="Run simulation match in CLI mode")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium",
                        help="AI difficulty level: easy (Depth 1, Memory 3), medium (Depth 2, Memory 5), hard (Depth 3, Memory 8)")
    parser.add_argument("--depth", type=int, default=None, help="Override AI Minimax search depth")
    parser.add_argument("--goal", "--win-streak", dest="goal", type=int, choices=[4, 5], default=5,
                        help="Win streak condition: 5-in-a-row (default) or 4-in-a-row")
    parser.add_argument("--max-turns", type=int, default=40, help="Maximum turns for CLI simulation match (default: 40)")
    args = parser.parse_args()

    if args.cli:
        run_cli_match(win_streak=args.goal, difficulty=args.difficulty, depth=args.depth, max_turns=args.max_turns)
    else:
        # Launch Pygame GUI
        from gui import GobbletGUI
        gui = GobbletGUI(win_streak=args.goal, difficulty=args.difficulty)
        if args.depth is not None:
            gui.ai_depth = args.depth
            gui.agent.max_depth = args.depth
            gui.telemetry.max_depth = args.depth
        gui.run()


if __name__ == "__main__":
    main()
