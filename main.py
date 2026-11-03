from game_engine import GameState, Move, PLAYER_BLUE, PLAYER_ORANGE, SIZE_SYMBOLS

def render_board_cli(state: GameState):
    """Draw a clean 5x5 ASCII grid in the terminal displaying top pieces and stack depth."""
    print("\n" + "=" * 55)
    print(f" TURN {state.turn_number}: Player {state.current_player}'s Turn")
    print("=" * 55)
    print("     0      1      2      3      4  ")
    print("  +------+------+------+------+------+")
    for r in range(5):
        row_str = f"{r} |"
        for c in range(5):
            stack = state.get_cell_stack(r, c)
            if not stack:
                cell_repr = "  .   "
            else:
                top = stack[-1]
                depth = len(stack)
                color = "B" if top.player == PLAYER_BLUE else "O"
                cell_repr = f" {color}{SIZE_SYMBOLS[top.size]}(d{depth})"
            row_str += cell_repr + "|"
        print(row_str)
        print("  +------+------+------+------+------+")
    
    # Print reserves
    b_res = state.reserves[PLAYER_BLUE]
    o_res = state.reserves[PLAYER_ORANGE]
    print(f"Blue Reserves  : L:{b_res[3]}  M:{b_res[2]}  S:{b_res[1]}")
    print(f"Orange Reserves: L:{o_res[3]}  M:{o_res[2]}  S:{o_res[1]}")
    print("=" * 55 + "\n")

import argparse
from ai_agent import MinimaxAgent

def run_cli_match(win_streak: int = 5, difficulty: str = "Medium", max_turns: int = 40):
    """Run an automated terminal simulation match between two AI agents."""
    state = GameState(win_streak=win_streak)
    blue_ai = MinimaxAgent(player=PLAYER_BLUE)
    orange_ai = MinimaxAgent(player=PLAYER_ORANGE)
    blue_ai.set_difficulty(difficulty)
    orange_ai.set_difficulty(difficulty)

    render_board_cli(state)
    while not state.game_over and state.turn_number <= max_turns:
        agent = blue_ai if state.current_player == PLAYER_BLUE else orange_ai
        move = agent.get_best_move(state)
        if not move:
            break
        print(f"Action executed: {move}")
        state.apply_move(move)
        render_board_cli(state)

    if state.game_over:
        print(f"MATCH FINISHED! Winner: Player {state.winner} ({state.win_reason})")
    else:
        print("Match reached turn limit without a winner.")


def main():
    parser = argparse.ArgumentParser(description="Gobblet Gobblers 5x5 — Team 6 AI")
    parser.add_argument("--cli", action="store_true", help="Run match in terminal CLI mode")
    parser.add_argument("--difficulty", default="Medium", choices=["Easy", "Medium", "Hard"], help="AI Difficulty Tier")
    parser.add_argument("--goal", type=int, default=5, choices=[4, 5], help="Win streak length (4 or 5)")
    args = parser.parse_args()

    if args.cli:
        run_cli_match(win_streak=args.goal, difficulty=args.difficulty)
    else:
        from gui import PygameGUI
        state = GameState(win_streak=args.goal)
        app = PygameGUI(state)
        app.difficulty_tier = args.difficulty
        app.ai_agent.set_difficulty(args.difficulty)
        app.run()


if __name__ == "__main__":
    main()
