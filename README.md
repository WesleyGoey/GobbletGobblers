# Gobblet Gobblers 5x5: AI Minimax with Bounded Rationality (Forgetting Memory)

An advanced digital implementation of Thierry Denoual's **Gobblet Gobblers** scaled to a **5x5 grid**, featuring an **Adversarial Search AI (Minimax + Alpha-Beta Pruning)** with **Bounded Rationality ("Forgetting Memory")** and a **Live Decision Visualization HUD**.

Built for **University Artificial Intelligence Course (IMT01307503) AFL 2 Project**.

---

## 1. Game Concept & Rules

Scaled up from the standard 3x3 edition to a strategic 5x5 grid:
- **Board**: 5x5 grid (25 cells). Each cell is a LIFO stack capable of holding up to 3 nested pieces (strictly descending sizes: Small -> Medium -> Large).
- **Pieces**: 12 pieces per player:
  - 4 Large (Size 3)
  - 4 Medium (Size 2)
  - 4 Small (Size 1)
- **Players**:
  - **Player 1 (Blue)**: Human
  - **Player 2 (Orange)**: AI Agent
- **Win Condition**: The first player to form **5 in a row** horizontally, vertically, or diagonally of visible topmost pieces wins immediately.
- **Actions**:
  1. **Introduce from Reserve**: Place a piece from your reserve onto an empty cell or over any smaller piece (friendly gobbling allowed).
  2. **Move on Board**: Lift an exposed (topmost) piece of your color and move it to another cell (empty or over a smaller piece).
- **The Russian Doll Gobble Rule**: Size 3 can gobble Size 2 or 1; Size 2 can gobble Size 1; Size 1 cannot gobble anything. Equal or larger pieces cannot be covered.
- **The Touch-Move Rule**: Selecting an active piece commits the player to moving it if legal moves exist.
- **The Uncover Trap**: Lifting a piece reveals the piece underneath. If lifting reveals an opponent's winning line, the active player must block it with the lifted piece or forfeit/lose immediately!

---

## 2. Artificial Intelligence Architecture

### A. Problem Formulation (PEAS)
- **Performance Measure**: Win with minimum turns (+10,000), prevent defeat (-10,000), maximize 4-of-5 and 3-of-5 line potential, control center cells `(2,2)` and inner ring, preserve Large piece reserve advantage, and avoid uncover traps.
- **Environment**: 5x5 grid, dynamic visible board layer + hidden buried stack layers, deterministic transition model, imperfect information under bounded memory.
- **Actuators**: Reserve drop actions `Drop(size, to_pos)` and reposition moves `Move(from_pos, to_pos)`.
- **Sensors**: Visible board state, reserve inventories, turn counter, and a finite temporal memory window of recent moves.

### B. Minimax with Alpha-Beta Pruning
The agent evaluates game trees up to depth 3 plies using minimax value propagation:
$$\alpha = \max(\alpha, \text{score}), \quad \beta = \min(\beta, \text{score})$$
Branches where $\alpha \ge \beta$ are pruned immediately, reducing search space from $O(b^d)$ to $O(b^{d/2})$ with optimal move ordering.

### C. Bounded Rationality: The "Forgetting Memory" Window (5 Moves)
To simulate human cognitive load and imperfect memory:
- The AI maintains a temporal memory window of **5 moves**.
- Buried pieces covered within the last 5 turns ($\Delta t \le 5$) are **remembered**.
- Any buried pieces covered more than 5 turns ago ($\Delta t > 5$) are **forgotten**.
- **Internal Belief State**: The AI constructs an imagined belief board for Minimax search where forgotten buried pieces are treated as `UNKNOWN / EMPTY` underneath!
- **Consequence**: If the AI lifts a piece covering a forgotten opponent piece, it will not anticipate the uncover trap during search — creating realistic, human-like oversights under pressure!

---

## 3. Real-Time Decision Visualization Panel

During the AI's turn, the GUI displays a live real-time HUD panel:
1. **Live Search Telemetry**: Current search depth, nodes visited, pruning cutoff count, elapsed time, and memory window indicator.
2. **Search Tree Branch Exploration**: The active branch path currently being traversed by Minimax (`Root -> Depth 1 -> Depth 2`).
3. **Candidate Move Scores & Cutoffs**: Real-time evaluation scores for every legal root move, with `[PRUNED]` tags on alpha-beta cutoffs and `★ [CHOSEN]` highlighted in neon green upon move selection.
4. **Memory Belief State Table**: Detailed breakdown of every stack on the board with buried pieces:
   - `Cell (r, c): Top: [Piece] | Buried: [Piece] [REMEMBERED, kt ago]`
   - `Cell (r, c): Top: [Piece] | Buried: UNKNOWN [FORGOTTEN >5t]`

---

## 4. Installation & Running

### Requirements
- Python 3.10+
- `pygame` 2.6+

### Install Dependencies
```bash
pip install pygame
```

### Launch Interactive GUI
```bash
python main.py
```

### Run Table 3 Empirical Benchmark (Pure Minimax vs Alpha-Beta Pruning)
```bash
python run_experiments.py
```

### Run Automated Unit & GUI Test Suite
```bash
python -m unittest discover -s tests
```

---

## 5. Controls & Shortcuts
- **Click Blue Reserve Piece**: Selects piece to introduce. Valid drop targets highlight in green on the board.
- **Click Board Piece**: Selects your visible piece to relocate. Valid targets highlight in green.
- **Click Target Cell**: Places the selected piece.
- **[RESET] New Game**: Resets the board and memory.
- **[GOAL] 5-in-a-row**: Displays the fixed 5-in-a-row win condition.
- **[DIFFICULTY] Difficulty Toggle**: Cycles AI difficulty between Easy (Depth 1, Memory 3), Medium (Depth 2, Memory 5), and Hard (Depth 3, Memory 8).