"""
run_experiments.py - Empirical Benchmarking Suite for Tabel 3.

Directly evaluates and demonstrates the computational difference between:
1. Pure Minimax (without pruning)
2. Minimax with Alpha-Beta Pruning

across representative tactical scenarios on the 5x5 Gobblet Gobblers board.
Measures:
- Nodes evaluated (Tree reduction efficiency)
- Wall-clock search latency in milliseconds (Execution speedup)
- Move identity / optimality preservation (100% equivalence)
"""

import os
import sys
import time
import argparse
import csv
from typing import List, Tuple, Dict, Any, Optional

from game_engine import GameState, Move, PLAYER_BLUE, PLAYER_ORANGE, PLAYER_NAMES
from ai_agent import MinimaxAgent


def get_default_scenarios(win_streak: int = 5) -> List[Tuple[str, GameState]]:
    """
    Serves as the benchmark scenario suite generator for Table 3 empirical evaluations.

    Constructs a standardized, reproducible collection of 5 distinct tactical board configurations
    spanning critical phases of a 5x5 Gobblet Gobblers match:
    1. Opening (Turn 2): Single center drop assessing early branching factor.
    2. Developed Center (Turn 6): Multiple distributed pieces evaluating mid-game spatial branching.
    3. Defensive Threat (3-in-a-row): Opponent line building requiring urgent defensive counter-actions.
    4. Gobble Battle (Turn 6): Repeated stacking/nesting on a single cell evaluating 3D stack tracking.
    5. Diagonal Threat (Turn 6): Multi-line diagonal alignment testing complex geometric pattern evaluation.

    Parameters:
        win_streak: Target consecutive alignment length (must be 5 for 5x5 rules).

    Returns:
        A list of (scenario_name, GameState) tuples ready for search algorithm benchmarking.

    Raises:
        ValueError: If win_streak is not 5.
    """
    if win_streak != 5:
        raise ValueError("Only 5-in-a-row is supported as a win condition on a 5x5 board")

    scenarios: List[Tuple[str, GameState]] = []

    # Scenario 1: Initial opening (Turn 2)
    s1 = GameState(win_streak=win_streak)
    s1.apply_move(Move(PLAYER_BLUE, 'drop', size=3, to_pos=(2, 2)))
    scenarios.append(("Opening (Turn 2)", s1))

    # Scenario 2: Developed center (Turn 6)
    s2 = GameState(win_streak=win_streak)
    s2.apply_move(Move(PLAYER_BLUE, 'drop', size=3, to_pos=(2, 2)))
    s2.apply_move(Move(PLAYER_ORANGE, 'drop', size=3, to_pos=(1, 1)))
    s2.apply_move(Move(PLAYER_BLUE, 'drop', size=2, to_pos=(2, 3)))
    s2.apply_move(Move(PLAYER_ORANGE, 'drop', size=2, to_pos=(1, 2)))
    s2.apply_move(Move(PLAYER_BLUE, 'drop', size=1, to_pos=(2, 4)))
    scenarios.append(("Developed Center (Turn 6)", s2))

    # Scenario 3: Defensive Threat (Opponent 3-in-a-row)
    s3 = GameState(win_streak=win_streak)
    s3.apply_move(Move(PLAYER_BLUE, 'drop', size=3, to_pos=(0, 0)))
    s3.apply_move(Move(PLAYER_ORANGE, 'drop', size=2, to_pos=(4, 4)))
    s3.apply_move(Move(PLAYER_BLUE, 'drop', size=3, to_pos=(0, 1)))
    s3.apply_move(Move(PLAYER_ORANGE, 'drop', size=2, to_pos=(4, 3)))
    s3.apply_move(Move(PLAYER_BLUE, 'drop', size=3, to_pos=(0, 2)))
    scenarios.append(("Defensive Threat (3-in-row)", s3))

    # Scenario 4: Midgame Gobble Battle
    s4 = GameState(win_streak=win_streak)
    s4.apply_move(Move(PLAYER_BLUE, 'drop', size=1, to_pos=(2, 2)))
    s4.apply_move(Move(PLAYER_ORANGE, 'drop', size=2, to_pos=(2, 2)))  # Gobbled
    s4.apply_move(Move(PLAYER_BLUE, 'drop', size=3, to_pos=(2, 2)))  # Gobbled again
    s4.apply_move(Move(PLAYER_ORANGE, 'drop', size=3, to_pos=(2, 1)))
    s4.apply_move(Move(PLAYER_BLUE, 'drop', size=2, to_pos=(1, 1)))
    scenarios.append(("Gobble Battle (Turn 6)", s4))

    # Scenario 5: Multiple Attack Lines
    s5 = GameState(win_streak=win_streak)
    s5.apply_move(Move(PLAYER_BLUE, 'drop', size=2, to_pos=(1, 1)))
    s5.apply_move(Move(PLAYER_ORANGE, 'drop', size=3, to_pos=(3, 3)))
    s5.apply_move(Move(PLAYER_BLUE, 'drop', size=2, to_pos=(2, 2)))
    s5.apply_move(Move(PLAYER_ORANGE, 'drop', size=3, to_pos=(3, 2)))
    s5.apply_move(Move(PLAYER_BLUE, 'drop', size=2, to_pos=(4, 4)))
    scenarios.append(("Diagonal Threat (Turn 6)", s5))

    return scenarios


def _format_diff_pct(val: float) -> str:
    """
    Serves as a specialized numerical difference formatter for Table 3 benchmark reports.

    Formats floating point percentage variations into standardized, aligned string tokens
    with explicit signed indicators ('-' for reductions/savings, '+' for increases),
    handling floating zero thresholds safely.

    Parameters:
        val: The percentage value to format (e.g., node reduction percentage or time delta).

    Returns:
        A formatted string such as '-89.6%', '+12.4%', or '0.0%'.
    """
    if abs(val) < 0.05:
        return "0.0%"
    if val > 0:
        return f"-{val:.1f}%"
    return f"+{abs(val):.1f}%"


def run_table_3_experiment(
    scenarios: Optional[List[Tuple[str, GameState]]] = None,
    depth: int = 2,
    memory_window: int = 5,
    min_reduction: Optional[float] = None,
    verbose: bool = True,
    print_markdown: bool = False,
    csv_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs the empirical benchmark for Table 3: Pure Minimax vs Alpha-Beta Pruning.

    Returns:
        Dict containing per-scenario metrics, aggregated totals, speedup, and pass status.
    """
    if scenarios is None:
        scenarios = get_default_scenarios()

    # Alpha-Beta cutoffs require at least 2 plies. Depth 1 evaluates all legal moves statically.
    if min_reduction is None:
        min_reduction = 30.0 if depth >= 2 else 0.0

    results: List[Dict[str, Any]] = []
    all_moves_matched = True
    total_raw_nodes = 0
    total_pruned_nodes = 0
    total_raw_time = 0.0
    total_ab_time = 0.0

    if verbose:
        print("\n" + "=" * 120)
        print(" TABEL 3: PERBANDINGAN EMPIRIS PURE MINIMAX VS ALPHA-BETA PRUNING (GOBBLET GOBBLERS 5x5)")
        print(f" Depth Pencarian: {depth} | Memory Window: {memory_window}")
        print("=" * 120)
        print(f"{'Skenario Papan':<28} | {'Pure Minimax':<18} | {'Alpha-Beta':<18} | {'Node Terpangkas':<15} | {'Hemat Waktu':<11} | {'Speedup':<8} | {'Langkah'}")
        print("-" * 120)

    for name, st in scenarios:
        active_player = st.current_player

        # 1. Pure Minimax (alpha-beta pruning disabled)
        agent_raw = MinimaxAgent(
            player=active_player,
            max_depth=depth,
            memory_window=memory_window,
            enable_pruning=False
        )
        agent_raw.step_delay = 0.0
        t0 = time.perf_counter()
        move_raw = agent_raw.get_action(st.clone())
        raw_time = (time.perf_counter() - t0) * 1000.0
        raw_nodes = agent_raw.nodes_visited

        # 2. Alpha-Beta Pruning (pruning enabled)
        agent_ab = MinimaxAgent(
            player=active_player,
            max_depth=depth,
            memory_window=memory_window,
            enable_pruning=True
        )
        agent_ab.step_delay = 0.0
        t1 = time.perf_counter()
        move_ab = agent_ab.get_action(st.clone())
        ab_time = (time.perf_counter() - t1) * 1000.0
        ab_nodes = agent_ab.nodes_visited

        nodes_cut = raw_nodes - ab_nodes
        reduction_pct = ((nodes_cut) / raw_nodes * 100.0) if raw_nodes > 0 else 0.0
        time_cut = raw_time - ab_time
        time_cut_pct = (time_cut / raw_time * 100.0) if raw_time > 0 else 0.0
        speedup = (raw_time / ab_time) if ab_time > 0 else 1.0

        is_same_move = (str(move_raw) == str(move_ab))
        if not is_same_move:
            all_moves_matched = False

        total_raw_nodes += raw_nodes
        total_pruned_nodes += ab_nodes
        total_raw_time += raw_time
        total_ab_time += ab_time

        item = {
            "name": name,
            "raw_nodes": raw_nodes,
            "ab_nodes": ab_nodes,
            "nodes_cut": nodes_cut,
            "reduction_pct": reduction_pct,
            "raw_time": raw_time,
            "ab_time": ab_time,
            "time_cut": time_cut,
            "time_cut_pct": time_cut_pct,
            "speedup": speedup,
            "same_move": is_same_move,
            "move_raw": str(move_raw),
            "move_ab": str(move_ab)
        }
        results.append(item)

        if verbose:
            raw_str = f"{raw_time:>6.1f}ms ({raw_nodes:>5}n)"
            ab_str = f"{ab_time:>6.1f}ms ({ab_nodes:>5}n)"
            cut_str = f"{_format_diff_pct(reduction_pct):>15}"
            time_str = f"{_format_diff_pct(time_cut_pct):>11}"
            sp_str = f"{speedup:>5.1f}x"
            mv_str = "100% Cocok" if is_same_move else "BEDA"
            print(f"{name:<28} | {raw_str:<18} | {ab_str:<18} | {cut_str} | {time_str} | {sp_str:<8} | {mv_str}")

    overall_nodes_cut = total_raw_nodes - total_pruned_nodes
    overall_reduction = ((overall_nodes_cut) / total_raw_nodes * 100.0) if total_raw_nodes > 0 else 0.0
    overall_time_cut = total_raw_time - total_ab_time
    overall_time_pct = (overall_time_cut / total_raw_time * 100.0) if total_raw_time > 0 else 0.0
    overall_speedup = (total_raw_time / total_ab_time) if total_ab_time > 0 else 1.0

    # Pruning benchmark passes if decisions match and node reduction satisfies threshold
    passed = all_moves_matched and (overall_reduction >= min_reduction)

    if verbose:
        print("-" * 120)
        tot_raw = f"{total_raw_time:>6.1f}ms ({total_raw_nodes:>5}n)"
        tot_ab = f"{total_ab_time:>6.1f}ms ({total_pruned_nodes:>5}n)"
        tot_cut = f"{_format_diff_pct(overall_reduction):>15}"
        tot_time = f"{_format_diff_pct(overall_time_pct):>11}"
        tot_sp = f"{overall_speedup:>5.1f}x"
        tot_mv = "COCOK" if all_moves_matched else "BEDA"
        print(f"{'TOTAL AKUMULASI':<28} | {tot_raw:<18} | {tot_ab:<18} | {tot_cut} | {tot_time} | {tot_sp:<8} | {tot_mv}")
        print("=" * 120)
        print(" Ringkasan Efisiensi Algoritma (Tabel 3):")
        print(f" • Penghematan Node (Search Space Cut) : {overall_reduction:.1f}% ({overall_nodes_cut:,} node dipangkas)")
        print(f" • Penghematan Waktu Komputasi        : {overall_time_pct:.1f}% ({overall_time_cut:,.1f}ms lebih cepat)")
        print(f" • Faktor Percepatan (Speedup Factor) : {overall_speedup:.1f}x lipat")
        print(f" • Konsistensi Keputusan (Move Match) : {'YA (100% Identik - Optimalitas Terjaga)' if all_moves_matched else 'TIDAK (Ada Deviasi Langkah)'}")
        print(f" • Status Hasil Pengujian             : {'BERHASIL' if passed else 'GAGAL'}")
        print("=" * 120)

    if print_markdown and verbose:
        print("\n### Format Markdown (Siap Salin ke Laporan / Paper):\n")
        print("| Skenario Papan | Pure Minimax (Waktu) | Pure Minimax (Node) | Alpha-Beta (Waktu) | Alpha-Beta (Node) | Hemat Node | Hemat Waktu | Speedup | Langkah Identik |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in results:
            print(f"| {r['name']} | {r['raw_time']:.1f} ms | {r['raw_nodes']:,} | {r['ab_time']:.1f} ms | {r['ab_nodes']:,} | {_format_diff_pct(r['reduction_pct'])} | {_format_diff_pct(r['time_cut_pct'])} | {r['speedup']:.1f}x | {'Ya' if r['same_move'] else 'Tidak'} |")
        print(f"| **TOTAL** | **{total_raw_time:.1f} ms** | **{total_raw_nodes:,}** | **{total_ab_time:.1f} ms** | **{total_pruned_nodes:,}** | **{_format_diff_pct(overall_reduction)}** | **{_format_diff_pct(overall_time_pct)}** | **{overall_speedup:.1f}x** | **{'100% Cocok' if all_moves_matched else 'Deviasi'}** |")
        print()

    if csv_path:
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Skenario", "Pure_Minimax_ms", "Pure_Minimax_nodes",
                "AlphaBeta_ms", "AlphaBeta_nodes", "Node_Reduction_Pct",
                "Time_Reduction_Pct", "Speedup", "Move_Pure_Minimax",
                "Move_Alpha_Beta", "Same_Move"
            ])
            for r in results:
                writer.writerow([
                    r["name"], f"{r['raw_time']:.2f}", r["raw_nodes"],
                    f"{r['ab_time']:.2f}", r["ab_nodes"], f"{r['reduction_pct']:.2f}",
                    f"{r['time_cut_pct']:.2f}", f"{r['speedup']:.2f}",
                    r["move_raw"], r["move_ab"], r["same_move"]
                ])
            writer.writerow([
                "TOTAL", f"{total_raw_time:.2f}", total_raw_nodes,
                f"{total_ab_time:.2f}", total_pruned_nodes, f"{overall_reduction:.2f}",
                f"{overall_time_pct:.2f}", f"{overall_speedup:.2f}",
                "-", "-", all_moves_matched
            ])
        if verbose:
            print(f"[OK] Data Tabel 3 berhasil diekspor ke: {csv_path}")

    return {
        "passed": passed,
        "scenarios": results,
        "total_raw_nodes": total_raw_nodes,
        "total_pruned_nodes": total_pruned_nodes,
        "overall_nodes_cut": overall_nodes_cut,
        "overall_reduction": overall_reduction,
        "total_raw_time": total_raw_time,
        "total_ab_time": total_ab_time,
        "overall_time_cut": overall_time_cut,
        "overall_time_pct": overall_time_pct,
        "overall_speedup": overall_speedup,
        "moves_matched": all_moves_matched
    }


# Backwards compatibility alias
test_3_alpha_beta_pruning = run_table_3_experiment


def main():
    """
    CLI entrypoint for running Table 3 empirical experiments comparing Pure Minimax vs Alpha-Beta Pruning.

    Parses command-line arguments (--depth, --memory, --markdown, --csv), validates parameter
    bounds, invokes run_table_3_experiment across standard test scenarios, and exits with 0 on pass
    or 1 on validation failure.
    """
    parser = argparse.ArgumentParser(
        description="Tabel 3: Pengujian Empiris Pure Minimax vs Alpha-Beta Pruning (Gobblet Gobblers 5x5)"
    )
    parser.add_argument("--depth", type=int, default=2, help="Kedalaman pencarian AI Minimax (default: 2, minimal: 1)")
    parser.add_argument("--memory", type=int, default=5, help="Ukuran temporal window memori AI (default: 5, minimal: 1)")
    parser.add_argument("--markdown", action="store_true", help="Cetak tabel dalam format Markdown untuk laporan")
    parser.add_argument("--csv", type=str, default=None, help="Simpan hasil tabel ke file CSV")
    args = parser.parse_args()

    if args.depth < 1:
        parser.error("--depth must be an integer >= 1")
    if args.memory < 1:
        parser.error("--memory must be an integer >= 1")

    results = run_table_3_experiment(
        depth=args.depth,
        memory_window=args.memory,
        print_markdown=args.markdown,
        csv_path=args.csv
    )

    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
