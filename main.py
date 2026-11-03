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
from ai_agent import MinimaxAgent, RandomPlayer


DIFFICULTY_MAP = {
    'easy': {'depth': 1, 'memory_window': 3},
    'medium': {'depth': 2, 'memory_window': 5},
    'hard': {'depth': 3, 'memory_window': 8},
}


def run_cli_match(
    win_streak: int = 5,
    difficulty: str = "medium",
    depth: Optional[int] = None,
    max_turns: int = 40,
    opponent: str = "random",
    matches: int = 1
):
    """
    Executes automated CLI simulation matches between AI agents or against a random baseline player.

    Simulates headless matches on a 5x5 board without launching the Pygame GUI. Supports tournament
    runs with win-rate statistics, ASCII board printing for single matches, and configurable search
    depth and bounded rationality memory windows.

    Parameters:
        win_streak: Target continuous pieces in-a-row required to win (must be 5).
        difficulty: AI difficulty preset ('easy', 'medium', or 'hard') mapped from DIFFICULTY_MAP.
        depth: Explicit search depth override. If None, uses depth from difficulty preset.
        max_turns: Upper turn cutoff limit per match before declaring a draw (default 40).
        opponent: 'random' for RandomPlayer (Pemain Acak baseline) or 'ai' for self-play Minimax AI.
        matches: Number of consecutive simulation matches to run in tournament mode.

    Raises:
        ValueError: If win_streak is not 5.
    """
    if win_streak != 5:
        raise ValueError("Only 5-in-a-row is supported as a win condition on a 5x5 board")

    diff = difficulty.lower() if isinstance(difficulty, str) and difficulty.lower() in DIFFICULTY_MAP else "medium"
    cfg = DIFFICULTY_MAP[diff]
    search_depth = depth if depth is not None else cfg['depth']
    mem_win = cfg['memory_window']

    print("=" * 60)
    print(f" Gobblet Gobblers 5x5 - CLI Simulation ({win_streak}-in-a-row)")
    print(f" Difficulty: {diff.upper()} (Depth {search_depth}, Memory Window {mem_win})")
    p1_desc = "Player 1 (Blue - Random Player / Pemain Acak)" if opponent == "random" else "Player 1 (Blue - Minimax AI)"
    print(f" {p1_desc} vs Player 2 (Orange - Minimax AI)")
    print("=" * 60)

    ai_wins = 0
    p1_wins = 0
    draws = 0

    for match_num in range(1, matches + 1):
        if matches > 1:
            print(f">>> Match {match_num}/{matches} in progress... <<<")
        state = GameState(win_streak=win_streak)
        if opponent == "random":
            blue_agent = RandomPlayer(player=PLAYER_BLUE)
        else:
            blue_agent = MinimaxAgent(player=PLAYER_BLUE, max_depth=search_depth, memory_window=mem_win)
            blue_agent.step_delay = 0

        orange_agent = MinimaxAgent(player=PLAYER_ORANGE, max_depth=search_depth, memory_window=mem_win)
        orange_agent.step_delay = 0

        while not state.game_over and state.turn_number <= max_turns:
            active_p = state.current_player
            agent = blue_agent if active_p == PLAYER_BLUE else orange_agent
            p_name = PLAYER_NAMES[active_p]

            if matches == 1:
                print(f"\n--- Turn {state.turn_number}: {p_name} ---")
            move = agent.get_action(state)
            if not move:
                if matches == 1:
                    print(f"No legal moves for {p_name}! Match ends.")
                break

            if matches == 1:
                print(f"Action: {move}")
            state.apply_move(move)

            if matches == 1:
                for r in range(5):
                    row_str = " | ".join(str(state.get_top_piece(r, c) or "  .  ") for c in range(5))
                    print(f"  [{row_str}]")

        if state.winner == PLAYER_ORANGE:
            ai_wins += 1
        elif state.winner == PLAYER_BLUE:
            p1_wins += 1
        else:
            draws += 1

        if matches == 1:
            print("\n" + "=" * 60)
            if state.game_over:
                print(f"GAME OVER! Winner: {PLAYER_NAMES[state.winner]}")
                print(f"Reason: {state.win_reason}")
            else:
                print("Match reached turn limit without a winner.")
            print("=" * 60)

    if matches > 1:
        print("\n" + "=" * 60)
        print(f" SIMULATION TOURNAMENT SUMMARY ({matches} Matches)")
        print(f" Orange (Minimax AI) Wins : {ai_wins} ({(ai_wins / matches) * 100:.1f}%)")
        print(f" Blue ({'Pemain Acak' if opponent == 'random' else 'AI'}) Wins  : {p1_wins} ({(p1_wins / matches) * 100:.1f}%)")
        print(f" Draws                    : {draws} ({(draws / matches) * 100:.1f}%)")
        print("=" * 60)


def main():
    """
    Main CLI entrypoint parsing command-line flags and launching GUI or CLI simulation mode.

    Flags:
        --cli: If specified, runs headless CLI simulation match(es); otherwise launches Pygame GUI.
        --difficulty: Selects AI difficulty preset ('easy', 'medium', 'hard').
        --depth: Optional integer override for Minimax search tree depth.
        --goal / --win-streak: Win condition alignment length (default 5).
        --max-turns: Turn limit for CLI matches.
        --opponent: Opponent mode for CLI ('random' or 'ai').
        --matches: Number of automated matches to run in CLI mode.
    """
    parser = argparse.ArgumentParser(description="5x5 Gobblet Gobblers with Minimax & Bounded Rationality")
    parser.add_argument("--cli", action="store_true", help="Run simulation match in CLI mode")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium",
                        help="AI difficulty level: easy (Depth 1, Memory 3), medium (Depth 2, Memory 5), hard (Depth 3, Memory 8)")
    parser.add_argument("--depth", type=int, default=None, help="Override AI Minimax search depth")
    parser.add_argument("--goal", "--win-streak", dest="goal", type=int, choices=[5], default=5,
                        help="Win streak condition: 5-in-a-row")
    parser.add_argument("--max-turns", type=int, default=40, help="Maximum turns for CLI simulation match (default: 40)")
    parser.add_argument("--opponent", choices=["random", "ai"], default="random",
                        help="Opponent type in CLI simulation: 'random' (Pemain Acak / Random Player baseline) or 'ai' (Minimax AI)")
    parser.add_argument("--matches", type=int, default=1,
                        help="Number of simulation matches to run in CLI mode (default: 1)")
    args = parser.parse_args()

    if args.cli:
        run_cli_match(
            win_streak=args.goal,
            difficulty=args.difficulty,
            depth=args.depth,
            max_turns=args.max_turns,
            opponent=args.opponent,
            matches=args.matches
        )
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
