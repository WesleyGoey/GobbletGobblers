"""
run_experiments.py - Automated Testing & Empirical Benchmarking Suite for Tabel 3.

Runs the 3 computational test plans defined in Table 3:
1. Aturan Game (Full Unit Test Suite: 49 tests)
2. AI vs Pemain Acak (50 Matches against Random Baseline)
3. Alpha-Beta Pruning Efficiency Benchmark (Nodes, Wall-Clock Time, & Move Equivalence)

Note: Test 4 (AI vs Manusia) is an interactive user evaluation conducted in GUI via `python3 main.py`.
"""

import sys
import time
import random
import unittest
from typing import List, Tuple, Dict, Any, Optional

from game_engine import (
    GameState, Piece, Move,
    PLAYER_BLUE, PLAYER_ORANGE, PLAYER_NAMES,
    SIZE_NAMES
)
from ai_agent import MinimaxAgent, RandomPlayer


def test_1_aturan_game() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(" [PENGUJIAN 1] ATURAN GAME (Gobble, Relokasi, Uncover Trap, Win)")
    print("=" * 70)
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    start_t = time.time()
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    elapsed = time.time() - start_t
    
    passed = result.wasSuccessful()
    total_tests = result.testsRun
    failed = len(result.failures) + len(result.errors)
    
    print(f"\nHasil: {'BERHASIL' if passed else 'GAGAL'}")
    print(f"Total Tes: {total_tests} | Lolos: {total_tests - failed} | Gagal: {failed}")
    print(f"Waktu Eksekusi: {elapsed:.3f}s")
    
    return {
        "passed": passed,
        "total": total_tests,
        "failed": failed,
        "elapsed": elapsed
    }


def test_2_ai_vs_random(num_matches: int = 50) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f" [PENGUJIAN 2] AI VS PEMAIN ACAK ({num_matches} Pertandingan)")
    print("=" * 70)
    
    ai_wins = 0
    random_wins = 0
    draws = 0
    illegal_moves = 0
    
    ai_player = PLAYER_ORANGE
    rand_player = PLAYER_BLUE
    
    start_t = time.time()
    for match_idx in range(1, num_matches + 1):
        state = GameState(win_streak=5)
        # Alternate first player
        first_p = PLAYER_BLUE if match_idx % 2 == 1 else PLAYER_ORANGE
        state.current_player = first_p
        
        ai_agent = MinimaxAgent(player=ai_player, max_depth=2, memory_window=5)
        ai_agent.step_delay = 0.0
        rand_agent = RandomPlayer(player=rand_player)
        
        turn = 0
        max_turns = 60
        while not state.game_over and turn < max_turns:
            curr_p = state.current_player
            agent = ai_agent if curr_p == ai_player else rand_agent
            
            legal_moves = state.get_legal_moves(curr_p)
            if not legal_moves:
                break

            move = agent.get_action(state)
            if not move:
                break
                
            # Verification: zero illegal moves
            if move not in legal_moves:
                illegal_moves += 1
                print(f"[ILLEGAL MOVE DETECTED] Match {match_idx}, Turn {turn}: {move}")
                break
                
            state.apply_move(move)
            turn += 1
            
        if state.winner == ai_player:
            ai_wins += 1
        elif state.winner == rand_player:
            random_wins += 1
        else:
            draws += 1
            
        if match_idx % 10 == 0 or match_idx == num_matches:
            print(f"  Match {match_idx:2d}/{num_matches}: AI Wins={ai_wins}, Random Wins={random_wins}, Draws={draws}")
            
    elapsed = time.time() - start_t
    win_rate = (ai_wins / num_matches) * 100.0
    passed = (win_rate >= 95.0) and (illegal_moves == 0)
    
    print(f"\nHasil: {'BERHASIL' if passed else 'GAGAL'}")
    print(f"AI Win Rate: {win_rate:.1f}% (Kriteria: >= 95%)")
    print(f"Illegal Moves: {illegal_moves} (Kriteria: 0)")
    print(f"Total Waktu: {elapsed:.2f}s ({elapsed/num_matches:.3f}s / match)")
    
    return {
        "passed": passed,
        "matches": num_matches,
        "ai_wins": ai_wins,
        "random_wins": random_wins,
        "draws": draws,
        "win_rate": win_rate,
        "illegal_moves": illegal_moves,
        "elapsed": elapsed
    }


def test_3_alpha_beta_pruning() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(" [PENGUJIAN 3] ALPHA-BETA PRUNING VS PURE MINIMAX (Node & Move Check)")
    print("=" * 70)
    
    # Generate 5 realistic benchmark boards
    scenarios = []
    
    # Scenario 1: Initial opening (Turn 2)
    s1 = GameState(win_streak=5)
    s1.apply_move(Move(PLAYER_BLUE, 'drop', size=3, to_pos=(2, 2)))
    scenarios.append(("Opening (Turn 2)", s1))
    
    # Scenario 2: Developed center (Turn 6)
    s2 = GameState(win_streak=5)
    s2.apply_move(Move(PLAYER_BLUE, 'drop', size=3, to_pos=(2, 2)))
    s2.apply_move(Move(PLAYER_ORANGE, 'drop', size=3, to_pos=(1, 1)))
    s2.apply_move(Move(PLAYER_BLUE, 'drop', size=2, to_pos=(2, 3)))
    s2.apply_move(Move(PLAYER_ORANGE, 'drop', size=2, to_pos=(1, 2)))
    s2.apply_move(Move(PLAYER_BLUE, 'drop', size=1, to_pos=(2, 4)))
    scenarios.append(("Developed Center (Turn 6)", s2))
    
    # Scenario 3: Defensive Threat (Opponent 3-in-a-row)
    s3 = GameState(win_streak=5)
    s3.apply_move(Move(PLAYER_BLUE, 'drop', size=3, to_pos=(0, 0)))
    s3.apply_move(Move(PLAYER_ORANGE, 'drop', size=2, to_pos=(4, 4)))
    s3.apply_move(Move(PLAYER_BLUE, 'drop', size=3, to_pos=(0, 1)))
    s3.apply_move(Move(PLAYER_ORANGE, 'drop', size=2, to_pos=(4, 3)))
    s3.apply_move(Move(PLAYER_BLUE, 'drop', size=3, to_pos=(0, 2)))
    scenarios.append(("Defensive Threat (3-in-row)", s3))
    
    # Scenario 4: Midgame Gobble Battle
    s4 = GameState(win_streak=5)
    s4.apply_move(Move(PLAYER_BLUE, 'drop', size=1, to_pos=(2, 2)))
    s4.apply_move(Move(PLAYER_ORANGE, 'drop', size=2, to_pos=(2, 2)))  # Gobbled
    s4.apply_move(Move(PLAYER_BLUE, 'drop', size=3, to_pos=(2, 2)))  # Gobbled again
    s4.apply_move(Move(PLAYER_ORANGE, 'drop', size=3, to_pos=(2, 1)))
    s4.apply_move(Move(PLAYER_BLUE, 'drop', size=2, to_pos=(1, 1)))
    scenarios.append(("Gobble Battle (Turn 6)", s4))
    
    # Scenario 5: Multiple Attack Lines
    s5 = GameState(win_streak=5)
    s5.apply_move(Move(PLAYER_BLUE, 'drop', size=2, to_pos=(1, 1)))
    s5.apply_move(Move(PLAYER_ORANGE, 'drop', size=3, to_pos=(3, 3)))
    s5.apply_move(Move(PLAYER_BLUE, 'drop', size=2, to_pos=(2, 2)))
    s5.apply_move(Move(PLAYER_ORANGE, 'drop', size=3, to_pos=(3, 2)))
    s5.apply_move(Move(PLAYER_BLUE, 'drop', size=2, to_pos=(4, 4)))
    scenarios.append(("Diagonal Threat (Turn 6)", s5))

    results = []
    all_moves_matched = True
    total_raw_nodes = 0
    total_pruned_nodes = 0
    
    print(f"{'Skenario':<26} | {'Pure Minimax':<16} | {'Alpha-Beta':<16} | {'Waktu Terpangkas':<18} | {'Speedup':<8} | {'Move'}")
    print("-" * 96)
    
    total_raw_time = 0.0
    total_ab_time = 0.0

    for name, st in scenarios:
        # 1. Pure Minimax (pruning disabled)
        agent_raw = MinimaxAgent(player=PLAYER_ORANGE, max_depth=2, memory_window=5, enable_pruning=False)
        agent_raw.step_delay = 0.0
        t0 = time.perf_counter()
        move_raw = agent_raw.get_action(st.clone())
        raw_time = (time.perf_counter() - t0) * 1000.0
        raw_nodes = agent_raw.telemetry.nodes_visited
        
        # 2. Alpha-Beta Pruning (pruning enabled)
        agent_ab = MinimaxAgent(player=PLAYER_ORANGE, max_depth=2, memory_window=5, enable_pruning=True)
        agent_ab.step_delay = 0.0
        t1 = time.perf_counter()
        move_ab = agent_ab.get_action(st.clone())
        ab_time = (time.perf_counter() - t1) * 1000.0
        ab_nodes = agent_ab.telemetry.nodes_visited
        
        reduction = ((raw_nodes - ab_nodes) / raw_nodes * 100.0) if raw_nodes > 0 else 0.0
        time_cut = raw_time - ab_time
        time_cut_pct = (time_cut / raw_time * 100.0) if raw_time > 0 else 0.0
        speedup = raw_time / ab_time if ab_time > 0 else 1.0

        is_same_move = (str(move_raw) == str(move_ab))
        if not is_same_move:
            all_moves_matched = False
            
        total_raw_nodes += raw_nodes
        total_pruned_nodes += ab_nodes
        total_raw_time += raw_time
        total_ab_time += ab_time
        
        results.append({
            "name": name,
            "raw_nodes": raw_nodes,
            "ab_nodes": ab_nodes,
            "reduction_pct": reduction,
            "raw_time": raw_time,
            "ab_time": ab_time,
            "time_cut": time_cut,
            "speedup": speedup,
            "same_move": is_same_move
        })
        
        print(f"{name:<26} | {raw_time:>6.1f}ms ({raw_nodes:>5}n) | {ab_time:>6.1f}ms ({ab_nodes:>5}n) | {time_cut:>6.1f}ms (-{time_cut_pct:4.1f}%) | {speedup:>5.1f}x   | {'100%' if is_same_move else 'MISMATCH'}")

    overall_reduction = ((total_raw_nodes - total_pruned_nodes) / total_raw_nodes * 100.0)
    overall_time_cut = total_raw_time - total_ab_time
    overall_time_pct = (overall_time_cut / total_raw_time * 100.0) if total_raw_time > 0 else 0.0
    overall_speedup = total_raw_time / total_ab_time if total_ab_time > 0 else 1.0
    passed = all_moves_matched and (overall_reduction > 30.0)
    
    print("-" * 96)
    print(f"TOTAL AKUMULASI            | {total_raw_time:>6.1f}ms ({total_raw_nodes:>5}n) | {total_ab_time:>6.1f}ms ({total_pruned_nodes:>5}n) | {overall_time_cut:>6.1f}ms (-{overall_time_pct:4.1f}%) | {overall_speedup:>5.1f}x   | {'COCOK' if all_moves_matched else 'BEDA'}")
    print(f"Rata-rata Hemat Waktu      : {overall_time_pct:.1f}% ({overall_speedup:.1f}x lebih cepat)")
    print(f"Rata-rata Hemat Node       : {overall_reduction:.1f}%")
    print(f"Semua Langkah Identik      : {'YA (100% Cocok)' if all_moves_matched else 'TIDAK'}")
    print(f"Hasil Pengujian            : {'BERHASIL' if passed else 'GAGAL'}")
    
    return {
        "passed": passed,
        "scenarios": results,
        "overall_reduction": overall_reduction,
        "moves_matched": all_moves_matched
    }


def main():
    print("=" * 70)
    print(" GOBBLET GOBBLERS 5x5 - EKSEKUSI PENGUJIAN OTOMATIS (TABEL 3)")
    print(" Tim 6 AI - Universitas Ciputra")
    print("=" * 70)
    
    t1 = test_1_aturan_game()
    t2 = test_2_ai_vs_random(num_matches=50)
    t3 = test_3_alpha_beta_pruning()
    
    print("\n" + "=" * 70)
    print(" RANGKUMAN HASIL AKHIR PENGUJIAN OTOMATIS TABEL 3")
    print("=" * 70)
    win_str = f"{t2['win_rate']:.1f}%"
    red_str = f"{t3['overall_reduction']:.1f}%"
    print(f" 1. Aturan Game       : {'BERHASIL (49/49 Unit Tests Lolos)' if t1['passed'] else 'GAGAL'}")
    print(f" 2. AI vs Pemain Acak : {'BERHASIL (Win Rate: ' + win_str + ', 0 Illegal)' if t2['passed'] else 'GAGAL'}")
    print(f" 3. Alpha-Beta Pruning: {'BERHASIL (' + red_str + ' Node Terpangkas, Langkah 100% Identik)' if t3['passed'] else 'GAGAL'}")
    print("=" * 70)
    print(" Catatan: Pengujian 4 (AI vs Manusia) dilakukan secara kualitatif")
    print(" oleh pemain manusia langsung melalui antarmuka GUI: 'python3 main.py'")
    print("=" * 70)


if __name__ == "__main__":
    main()
