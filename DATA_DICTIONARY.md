# Data dictionary

## Source files (real)
| Item | Type | Real / generated | Notes |
|---|---|---|---|
| MNIST images (4 gzip files, ~11 MiB) | 28x28 uint8, 0-255 | **real** | http://yann.lecun.com/exdb/mnist/ (mirrors in `make_data.py`; md5 checked) |
| MNIST labels | uint8, 0-9 | **real** | digit 0 is dropped, labels stored as digit-1 (0..8) |

## `data/digits19.npz` (produced by `make_data.py`, seed 0)
| Key | Shape | Real / generated | Meaning |
|---|---|---|---|
| `x_train`, `y_train` | (~45k,28,28), (~45k,) | real | 50,000 random (seeded) training images, digit 0 removed |
| `x_val`, `y_val` | (~9k,28,28), (~9k,) | real | the other 10,000 training images, digit 0 removed (used for early checks + temperature scaling) |
| `x_test`, `y_test` | (~9k,28,28), (~9k,) | real | official 10,000 test images, digit 0 removed; writers differ from training |
| `source`, `seed` | str, int | metadata | `MNIST` or `OFFLINE sklearn digits (smoke test only)` |
Pixels are scaled to 0-1 at load time (`p7/data.py`).

## Generated objects (all seeded)
| Object | How it is made |
|---|---|
| Full Sudoku grid (81 ints, 1-9) | randomised backtracking with a seeded RNG (`generator.random_full_grid`) |
| Puzzle (81 ints, 0 = blank) | cells removed in random order while the solution stays **unique** (solver counts up to 2 solutions) until the target number of givens (25/30/35) remains |
| Mosaic (252x252 float) | 81 cells of 28x28. Given cell with digit d -> a real MNIST test image of digit d. Blank cell -> uniform noise in [0, 0.03]. **Not a photograph of a real Sudoku.** |
| Easy images | test images the trained network classifies correctly (used for all non-injected cells) |
| Hard images (injected misreads) | test images of digit d that the network gets **wrong** with d inside its top 3. If fewer than 3 exist, the images with the lowest p(d) are added (the demo/experiments report the *actual* misread count) |
| Misread level k | k randomly chosen given cells receive a hard image |
| Reading | blank if max pixel < 0.3 (brightness rule, no NN); else NN arg-max digit, `probs` = 9 temperature-scaled probabilities |

## Search / log objects
| Field | Meaning |
|---|---|
| correction log row | `cell` (0-80), `row`, `col`, `read` digit, `new` digit, `cost = ln p(read) - ln p(new)` (>= 0) |
| `nodes_expanded` / `nodes_generated` | popped-and-expanded / pushed search nodes |
| CSP `assignments`, `backtracks`, `revisions`, `prunings` | decisions made, assignments undone, AC-3 revise calls, domain values removed |

## Result CSVs (`results/`)
`exp*_raw.csv`: one row per (seed, puzzle, level, method). `exp*_summary.csv`/`.md`: mean and std across seeds
of the per-seed mean. `nn_metrics.json`: test accuracy / NLL / ECE for MLP (raw, scaled) and logistic regression. `confusion_matrix.csv`: MLP, rows = true digit-1, cols = predicted digit-1.
