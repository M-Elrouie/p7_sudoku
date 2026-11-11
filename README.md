# P7 - Handwritten Sudoku Reader and Solver (NN + Search + CSP)

A network reads a Sudoku written in handwritten digits; when it misreads, **A\*/UCS/Greedy search**
finds the cheapest set of corrections (cost = loss of probability), and a **CSP solver**
(Backtracking, Forward Checking, AC-3 - switchable) tests each corrected puzzle. If nothing works
within 3 corrections the system prints `UNREADABLE` and lists the least confident cells.

## Setup (Python 3.10+)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Exact run commands
```bash
python make_data.py                 # download + preprocess MNIST (seed 0)  -> data/digits19.npz
python train_nn.py --epochs 10      # MLP + logistic-regression baseline, temperature scaling
                                    # -> data/mlp.npz, results/nn_metrics.json, results/confusion_matrix.csv
python -m pytest -q                 # 32 tests (normal / edge / impossible)
python run_demo.py                  # integrated demo (2 misreads)      -> results/demo_*.png
python run_demo.py --impossible     # 5 misreads > limit 3 -> "UNREADABLE" + least confident cells
python run_demo.py --goal solvable  # minimum-version goal instead of "exactly one solution"
python run_experiments.py                   # all 3 experiments, 5 seeds x 100 puzzles/level (~3-10 min)
python run_experiments.py --goal unique     # same with the uniqueness goal (extension)
python run_experiments.py --quick           # smoke run
```
No internet? `python make_data.py --offline-digits` builds a **smoke-test-only** dataset from scikit-learn's
bundled 8x8 digits. Never report numbers from it.

## Layout
| Path | Purpose |
|---|---|
| `p7/data.py`, `make_data.py` | dataset loading / reproducible preprocessing, brightness blank rule |
| `p7/nn.py`, `train_nn.py` | numpy MLP + softmax regression baseline, ECE, NLL, confusion matrix, temperature scaling |
| `p7/csp.py` | Sudoku CSP: BT, FC, AC-3 (separate switches) + counters |
| `p7/search.py` | correction search: UCS / Greedy / A*, admissible heuristic, counters, trace |
| `p7/generator.py` | seeded puzzle generator (unique solution), mosaic builder, easy/hard image pools |
| `p7/integration.py` | read -> AC-3 -> search -> CSP pipeline; modes `nn` / `none` / `random` |
| `p7/eval.py`, `run_experiments.py` | the 3 experiments, tables, plots |
| `tests/` | 32 tests |

## Design notes (for the report)
* **Cost**: `ln p(read) - ln p(new)`; read = arg-max so cost >= 0 (UCS/A* need non-negative edges).
* **Search space**: a correction *set*, each cell at most once, generated in increasing cell order so every set appears once (no closed list, A* stays optimal with an admissible h). Candidate cells = the 12 least-confident given cells, 2 alternatives each, max 3 corrections.
* **A\* heuristic (admissible)**: a (unit, digit) group with k equal digits needs >= k-1 changed cells. Take a greedy packing of cell-disjoint groups; per group add the (k-1) cheapest per-cell minimum change costs; a node needing more changes than corrections left (or unchangeable cells) is a dead end. Disjoint groups cannot share a changed cell, so the sum never overestimates.
* **Goal test**: duplicate givens are rejected cheaply (this is exactly what root AC-3 detects); otherwise the full CSP runs. `--goal unique` requires exactly one solution (solver stops at 2).
* **Why uniqueness matters**: a misread can leave a puzzle solvable but *wrong*. In a quick check on the offline data, "solvable" accepted wrong solutions in 7/81 misread cases vs 1/81 with "unique".
* **AC-3 in the CSP**: `ac3_on=True` runs AC-3 at the root and again after each assignment. For `!=` constraints it subsumes FC and also propagates chains of forced cells.
* **Experiment protocol**: seeds change puzzles and images, not the trained network. Normal (non-injected) cells use correctly classified images, so the *injected level* is the controlled misread level; the **actual** misread count is logged. Level 4 exceeds the 3-correction limit on purpose. Random mode = 300 random correction sets (random cells, random digits). Exp 2 caps plain-BT at 100k assignments and reports the abort rate.


