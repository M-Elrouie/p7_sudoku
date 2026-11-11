# P7 - Handwritten Sudoku Reader and Solver (NN + Search + CSP)

A neural network reads a Sudoku written in handwritten digits. When it misreads a digit, a search finds the cheapest set of corrections (UCS, Greedy or A*), and a CSP solver (Backtracking, Forward Checking, AC-3 - each switchable) tests every corrected puzzle. If no solvable reading is found within 3 corrections, the system prints `UNREADABLE` and lists the least confident cells.

Course: Intelligent systems | Name: [Mohamed Ahmed Saed] | Report: [docs/P7_Sudoku_Report.docx]

## How the three parts cooperate
1. **NN** classifies each handwritten cell (digits 1-9) and outputs probabilities. Blank cells are detected by a brightness rule, not by the NN.
2. **AC-3** checks the first reading. Two equal digits in one row, column or box empty a domain, so the reading is "broken".
3. **Search** tries correction sets. State = corrections made so far, action = change one uncertain cell to one of its top alternatives, cost = ln p(read) - ln p(new) (never negative), limit = 3 corrections.
4. **CSP** tests each corrected puzzle. The first one that is solvable (or has exactly one solution, with `--goal unique`) wins.

## Setup (Python 3.10+)
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Mac / Linux
pip install -r requirements.txt
```

## Run commands (in this order)
```bash
python make_data.py                  # download + preprocess MNIST (seed 0) -> data/digits19.npz
python train_nn.py --epochs 10       # MLP + logistic-regression baseline   -> data/*.npz, results/nn_metrics.json,
                                     #                                         results/confusion_matrix.csv
python -m pytest -q                  # 32 tests: normal, edge and impossible cases
python run_demo.py                   # integrated demo (2 misreads)         -> results/demo_*.png
python run_demo.py --impossible      # 5 misreads > limit of 3 -> prints UNREADABLE + least confident cells
python run_experiments.py            # 3 experiments, goal "solvable"       -> results/
python run_experiments.py --goal unique --out results_unique    # goal "exactly one solution" -> results_unique/
```

Useful options: `run_demo.py --misreads 3 --seed 5 --givens 30 --goal solvable`,
`run_experiments.py --quick` (smoke run), `--exp 1 2 3`, `--seeds 5`, `--puzzles 100`.

Notes
* `make_data.py` tries three mirrors in order. On some Windows installs the first mirror prints an `SSL: CERTIFICATE_VERIFY_FAILED` error; the script then falls back to the next one. This is harmless. Check that no line starts with `WARNING: md5`.
* If every mirror fails, put the four MNIST `.gz` files in `data/raw/` by hand (names as in `make_data.py`) and rerun.
* `python make_data.py --offline-digits` builds a **smoke-test-only** dataset from scikit-learn's tiny digits. Never report numbers from it.
* The demo always writes to the same two image files. Copy them to new names if you want to keep several runs.
* A full experiments run takes roughly 5-15 minutes.

## Where the results are
| Folder | Contents |
|---|---|
| `results/` | experiments with goal **solvable**, NN metrics, confusion matrix, demo images |
| `results_unique/` | the same experiments with goal **unique** (exactly one solution) |

Each experiment writes `expN_raw.csv` (one row per run), `expN_summary.csv` and `expN_summary.md`
(mean ± std across 5 seeds) and its plots. See `DATA_DICTIONARY.md` for every field.

## Headline results (real MNIST, 5 seeds)
* **NN:** MLP 97.63 % test accuracy vs 91.98 % for logistic regression; ECE 0.0027 (already well calibrated, so temperature scaling, T = 1.04, did not help).
* **Exp 1 (UCS / Greedy / A*):** A* and UCS find the same optimal corrections; A* expands far fewer nodes. Exact-solution recovery of A* with 1 / 2 / 3 misreads: 0.844 / 0.678 / 0.544 with goal *solvable*, 0.956 / 0.872 / 0.802 with goal *unique*. Level 4 exceeds the 3-correction limit and recovers 0 by design.
* **Exp 2 (CSP):** BT > BT+FC > BT+FC+AC-3 in assignments at every difficulty (at 25 givens: about 33,000 / 9,700 / 80). Plain BT hit its 100,000-assignment cap on 13 % of 25-given puzzles. AC-3 does thousands of cheap revisions per puzzle, so on easy puzzles (35 givens) it is slower than FC.
* **Exp 3 (NN-guided vs none vs random):** NN-guided recovers 0.956 / 0.872 / 0.802; random correction (300 random sets) recovers 0.298 / 0.004 / 0.000; no correction recovers 0 whenever a cell is misread.
* **Main failure mode:** a *silent wrong answer*, where the corrected puzzle is solvable but not the true one. With goal *solvable* this happened in 73 / 156 / 213 of 500 puzzles (1 / 2 / 3 misreads); with goal *unique* in 16 / 59 / 83. The system said "unreadable" in only 5-16 of 500.

## Project layout
| Path | Purpose |
|---|---|
| `make_data.py`, `p7/data.py` | reproducible download + preprocessing, loading, blank-cell brightness rule |
| `train_nn.py`, `p7/nn.py` | numpy MLP and softmax-regression baseline, accuracy / NLL / ECE, confusion matrix, temperature scaling |
| `p7/csp.py` | Sudoku CSP: Backtracking, Forward Checking, AC-3 as separate switches, with counters |
| `p7/search.py` | correction search: UCS / Greedy / A*, admissible heuristic, counters, trace |
| `p7/generator.py` | seeded puzzle generator (unique solution), mosaic builder, easy / hard image pools |
| `p7/integration.py` | read -> AC-3 -> search -> CSP pipeline; modes `nn`, `none`, `random` |
| `p7/eval.py`, `run_experiments.py` | the three experiments, tables and plots |
| `run_demo.py` | integrated demo |
| `tests/` | 32 tests |
| `DATA_DICTIONARY.md` | every data field, real vs generated |

## Design notes
* **Cost:** ln p(read) - ln p(new). The read digit is the arg-max, so cost >= 0, which UCS and A* need.
* **Search space:** a correction *set*, each cell changed at most once, generated in increasing cell order so every set appears exactly once. Candidates = the 12 least confident given cells, 2 alternatives each, at most 3 corrections.
* **A\* heuristic (admissible):** a (unit, digit) group with k equal digits needs at least k-1 changed cells. Take a greedy packing of cell-disjoint groups and add, per group, the k-1 cheapest per-cell minimum change costs. Disjoint groups cannot share a changed cell, so the sum never overestimates. A node that needs more changes than corrections left is a dead end.
* **Goal test:** duplicate digits are rejected cheaply (this is what root AC-3 detects); otherwise the full CSP runs. `--goal unique` needs exactly one solution (the solver stops counting at 2).
* **AC-3 in the CSP:** runs at the root and again after every assignment. For "different" constraints it subsumes FC and also propagates chains of forced cells.
* **Experiment protocol:** seeds change the puzzles and images, not the trained network. Non-injected cells use correctly classified images, so the injected level equals the real number of misreads (checked in `actual_misreads`). Random mode = 300 random correction sets. Exp 2 caps plain BT at 100,000 assignments and reports the abort rate.

## Known limitations
* Cost-based correction can return a confident wrong answer (see main failure mode above); the uniqueness goal reduces this but does not remove it.
* At most 3 corrections, so 4 or more misreads are reported as `UNREADABLE`.
* The sudokus are generated mosaics of MNIST digits, not photographs of real handwritten puzzles.