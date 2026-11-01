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
