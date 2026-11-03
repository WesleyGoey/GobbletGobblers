"""
gui_telemetry.py - Live Real-Time AI Decision Telemetry HUD for Gobblet Gobblers.

Renders:
1. AI Search & Bounded Memory Status Banner
2. Real-time Search Tree & Branch Trail with Pruning cutoffs
3. Candidate Moves with evaluated scores, [EVAL] active markers, and [CHOSEN] highlights
4. Memory Belief Table (What AI Remembers vs Forgot across 5-move bounded horizon)
"""

import pygame
import math
import time
from typing import Any, Tuple, Dict
from gui_renderer import TextSurfaceCache
from game_engine import PLAYER_BLUE


class TelemetryPanelRenderer:
    """Renders the AI Decision Telemetry and Cognitive Memory HUD panel."""

    def __init__(self, gui: Any):
        self.gui = gui
        self.text_cache = TextSurfaceCache(max_entries=300)

    def draw_visualization_panel(self):
        gui = self.gui
        screen = gui.screen
        x = gui.panel_rect.x + 16
        y = gui.panel_rect.y + 16

        # Outer panel frame
        pygame.draw.rect(screen, gui.COLOR_PANEL_BG, gui.panel_rect, border_radius=12)
        pygame.draw.rect(screen, gui.COLOR_BORDER, gui.panel_rect, 2, border_radius=12)

        # --- SECTION 1: HEADER & TELEMETRY ---
        title = self.text_cache.render(gui.font_header, "AI DECISION TELEMETRY (LIVE)", gui.COLOR_TEXT_HIGHLIGHT)
        screen.blit(title, (x, y))

        status_text = "THINKING..." if gui.ai_thinking else "IDLE (Awaiting Human)"
        status_col = gui.COLOR_ORANGE_LIGHT if gui.ai_thinking else gui.COLOR_GREEN_ACCENT
        status_surf = self.text_cache.render(gui.font_bold, status_text, status_col)
        pill_w = status_surf.get_width() + 26
        pill_rect = pygame.Rect(gui.panel_rect.right - pill_w - 20, y, pill_w, 24)
        pygame.draw.rect(screen, (22, 28, 38), pill_rect, border_radius=12)
        pygame.draw.rect(screen, status_col, pill_rect, 1, border_radius=12)
        pygame.draw.circle(screen, status_col, (pill_rect.x + 10, pill_rect.centery), 4)
        screen.blit(status_surf, (pill_rect.x + 18, pill_rect.centery - status_surf.get_height() // 2))
        y += 32

        # Telemetry metrics bar
        metrics_rect = pygame.Rect(x, y, gui.panel_rect.width - 32, 54)
        pygame.draw.rect(screen, gui.COLOR_CARD_BG, metrics_rect, border_radius=6)
        pygame.draw.rect(screen, gui.COLOR_BORDER, metrics_rect, 1, border_radius=6)

        m_line1 = f"Depth: {gui.telemetry.max_depth} plies  |  Nodes Visited: {gui.telemetry.nodes_visited}  |  Pruned: {gui.telemetry.prune_count}"
        t1 = self.text_cache.render(gui.font_mono, m_line1, gui.COLOR_TEXT_WHITE)
        metrics_rect_x = metrics_rect.x + 10
        screen.blit(t1, (metrics_rect_x, metrics_rect.y + 8))

        best_score_str = f"{gui.telemetry.best_score:+.1f}" if gui.telemetry.best_score != -math.inf else "N/A"
        time_str = f"{gui.telemetry.elapsed_time:.2f}s" if not gui.ai_thinking else f"{time.time() - gui.telemetry.start_time:.2f}s"
        m_line2 = f"Best Score: {best_score_str}  |  Time: {time_str}  |  Memory Window: {gui.agent.memory_window} turns"
        t2 = self.text_cache.render(gui.font_mono, m_line2, gui.COLOR_TEXT_MUTED)
        screen.blit(t2, (metrics_rect_x, metrics_rect.y + 30))
        y += 66

        # --- SECTION 2: LIVE SEARCH TREE BRANCH EXPLORATION ---
        sec2_title = self.text_cache.render(gui.font_bold, "SEARCH TREE PATH EXPLORATION", gui.COLOR_TEXT_WHITE)
        screen.blit(sec2_title, (x, y))
        y += 20

        tree_rect = pygame.Rect(x, y, gui.panel_rect.width - 32, 105)
        pygame.draw.rect(screen, (20, 24, 32), tree_rect, border_radius=6)
        pygame.draw.rect(screen, gui.COLOR_BORDER, tree_rect, 1, border_radius=6)

        ty = tree_rect.y + 8
        curr_branch = gui.telemetry.get_current_branch_copy()
        if curr_branch:
            for idx, step_desc in enumerate(curr_branch[:4]):
                prefix = "  " * idx + ("\\-- " if idx > 0 else "Root: ")
                txt_col = (100, 200, 255) if idx == 0 else gui.COLOR_TEXT_MUTED
                txt = self.text_cache.render(gui.font_mono, f"{prefix}{step_desc}", txt_col)
                screen.blit(txt, (tree_rect.x + 8, ty))
                ty += 22
        else:
            txt = self.text_cache.render(gui.font_mono, "Awaiting next AI search invocation...", gui.COLOR_TEXT_MUTED)
            screen.blit(txt, (tree_rect.x + 8, ty))

        # Show latest pruning cutoff event
        pruned_list = gui.telemetry.get_pruned_branches_copy()
        if pruned_list:
            latest_prune = pruned_list[-1]
            p_txt = self.text_cache.render(gui.font_small, latest_prune, gui.COLOR_PRUNED)
            screen.blit(p_txt, (tree_rect.x + 8, tree_rect.bottom - 20))

        y += 115

        # --- SECTION 3: CANDIDATE MOVES & SCORES ---
        sec3_title = self.text_cache.render(gui.font_bold, "CANDIDATE MOVES & MINIMAX SCORES", gui.COLOR_TEXT_WHITE)
        screen.blit(sec3_title, (x, y))
        y += 20

        cand_rect = pygame.Rect(x, y, gui.panel_rect.width - 32, 220)
        pygame.draw.rect(screen, (20, 24, 32), cand_rect, border_radius=6)
        pygame.draw.rect(screen, gui.COLOR_BORDER, cand_rect, 1, border_radius=6)

        cy = cand_rect.y + 8
        items = gui.telemetry.get_candidate_scores_copy()
        items.sort(key=lambda kv: kv[1], reverse=True)

        chosen_str = str(gui.telemetry.best_move) if gui.telemetry.best_move else ""

        if not items:
            empty_msg = self.text_cache.render(gui.font_mono, "No evaluated moves yet.", gui.COLOR_TEXT_MUTED)
            screen.blit(empty_msg, (cand_rect.x + 10, cy))
        else:
            for move_str, score in items[:8]:  # Top 8 candidate moves
                is_best = (move_str == chosen_str)
                is_active = (curr_branch and move_str == curr_branch[0])

                if is_best and not gui.ai_thinking:
                    prefix = "[CHOSEN] "
                    text_col = gui.COLOR_GREEN_ACCENT
                    bg_col = (25, 55, 35)
                elif is_active:
                    prefix = "[EVAL]   "
                    text_col = (100, 220, 255)
                    bg_col = (30, 45, 65)
                else:
                    prefix = "[CAND]   "
                    text_col = gui.COLOR_TEXT_WHITE
                    bg_col = None

                line_rect = pygame.Rect(cand_rect.x + 4, cy - 2, cand_rect.width - 8, 22)
                if bg_col:
                    pygame.draw.rect(screen, bg_col, line_rect, border_radius=4)

                line_text = f"{prefix}{move_str:<32} Score: {score:+.1f}"
                c_surf = self.text_cache.render(gui.font_mono, line_text, text_col)
                screen.blit(c_surf, (cand_rect.x + 8, cy))
                cy += 24

        y += 230

        # --- SECTION 4: AI COGNITIVE MEMORY HUD ---
        sec4_title = self.text_cache.render(gui.font_bold, "AI COGNITIVE MEMORY HUD", gui.COLOR_TEXT_WHITE)
        screen.blit(sec4_title, (x, y))

        mem_win = gui.agent.memory_window
        hud_badge = self.text_cache.render(
            gui.font_small,
            f"BUFFER: {mem_win} MOVES  |  BOUNDED HORIZON",
            gui.COLOR_GREEN_ACCENT
        )
        screen.blit(hud_badge, (gui.panel_rect.right - hud_badge.get_width() - 20, y + 2))
        y += 20

        mem_rect = pygame.Rect(x, y, gui.panel_rect.width - 32, gui.panel_rect.bottom - y - 16)
        pygame.draw.rect(screen, (20, 24, 32), mem_rect, border_radius=6)
        pygame.draw.rect(screen, gui.COLOR_BORDER, mem_rect, 1, border_radius=6)

        my = mem_rect.y + 10

        note1 = self.text_cache.render(
            gui.font_small,
            f"Episodic Memory Horizon: AI retains move provenance for up to {mem_win} turns.",
            (130, 165, 205)
        )
        screen.blit(note1, (mem_rect.x + 10, my))
        my += 16
        note2 = self.text_cache.render(
            gui.font_small,
            f"Moves older than {mem_win} turns decay from working memory. Board under-layers are never X-rayed.",
            (100, 130, 160)
        )
        screen.blit(note2, (mem_rect.x + 10, my))
        my += 20

        # Capacity gauge / buffer slots
        curr_turn = gui.state.turn_number
        recent_moves = gui.state.move_history[-mem_win:] if gui.state.move_history else []
        slots_used = len(recent_moves)

        slot_w = (mem_rect.width - 20 - (mem_win - 1) * 6) // mem_win
        for s_idx in range(mem_win):
            slot_x = mem_rect.x + 10 + s_idx * (slot_w + 6)
            slot_rect = pygame.Rect(slot_x, my, slot_w, 20)
            is_filled = (s_idx < slots_used)
            s_bg = (30, 52, 42) if is_filled else (26, 30, 40)
            s_border = (46, 204, 113) if is_filled else (45, 52, 68)
            pygame.draw.rect(screen, s_bg, slot_rect, border_radius=4)
            pygame.draw.rect(screen, s_border, slot_rect, 1, border_radius=4)

            s_text = f"Slot {s_idx + 1}" if not is_filled else f"M-{slots_used - 1 - s_idx}"
            s_col = gui.COLOR_GREEN_ACCENT if is_filled else gui.COLOR_TEXT_MUTED
            s_surf = self.text_cache.render(gui.font_small, s_text, s_col)
            screen.blit(s_surf, (slot_rect.centerx - s_surf.get_width() // 2, slot_rect.centery - s_surf.get_height() // 2))

        my += 26

        # Recent action history queue
        if not recent_moves:
            no_moves = self.text_cache.render(
                gui.font_mono,
                f"Game start: Working memory buffer ready. 0/{mem_win} slots active.",
                gui.COLOR_TEXT_MUTED
            )
            screen.blit(no_moves, (mem_rect.x + 10, my + 15))
        else:
            active_threshold = max(2, mem_win // 2)
            for t_num, mv, desc in reversed(recent_moves):
                turns_ago = curr_turn - t_num
                if turns_ago <= 1:
                    status_col = (100, 220, 255)
                    age_badge = f"[T-{turns_ago} fresh]"
                elif turns_ago <= active_threshold:
                    status_col = gui.COLOR_GREEN_ACCENT
                    age_badge = f"[T-{turns_ago} active]"
                elif turns_ago <= mem_win:
                    status_col = gui.COLOR_TEXT_HIGHLIGHT
                    age_badge = f"[T-{turns_ago} decaying]"
                else:
                    status_col = gui.COLOR_FORGOTTEN
                    age_badge = f"[T-{turns_ago} expired]"

                row_rect = pygame.Rect(mem_rect.x + 8, my, mem_rect.width - 16, 21)
                p_bg = (24, 38, 52) if mv.player == PLAYER_BLUE else (48, 34, 24)
                pygame.draw.rect(screen, p_bg, row_rect, border_radius=4)

                m_surf = self.text_cache.render(gui.font_mono, f"Turn {t_num:<2} {desc:<32}", gui.COLOR_TEXT_WHITE)
                screen.blit(m_surf, (row_rect.x + 8, row_rect.y + 2))

                stat_surf = self.text_cache.render(gui.font_bold, age_badge, status_col)
                screen.blit(stat_surf, (row_rect.right - stat_surf.get_width() - 8, row_rect.y + 2))

                my += 24
