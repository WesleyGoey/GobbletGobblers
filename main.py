"""
main.py - Entry point for 5x5 Gobblet Gobblers.

Usage:
  python main.py    # Launches full Pygame GUI with interactive board & live AI panel
"""

from gui import GobbletGUI


def main():
    """
    Main application entry point. Launches the interactive Gobblet Gobblers GUI.
    """
    gui = GobbletGUI(win_streak=5, difficulty="medium")
    gui.run()


if __name__ == "__main__":
    main()
