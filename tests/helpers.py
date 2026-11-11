import math
import numpy as np
from p7.generator import make_puzzle
from p7.csp import PEERS

KNOWN = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
KNOWN_SOL = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"


def digits(s):
    return [int(c) for c in s]


def synthetic_case(seed, n_mis, givens=30, conf=0.97, read_p=0.52, true_p=0.45):
    """Controlled misreads. Returns (grid0, probs, puzzle, solution, misread cells, extra margin).
    Each misread cell is read as a digit that DUPLICATES a peer given (so the puzzle is broken)."""
    rng = np.random.default_rng(seed)
    puz, sol = make_puzzle(rng, givens)
    gv = [i for i in range(81) if puz[i]]
    probs = np.full((81, 9), 1 / 9)
    grid0 = list(puz)
    for i in gv:
        p = np.full(9, (1 - conf) / 8); p[puz[i] - 1] = conf
        probs[i] = p
    mis = []
    for i in rng.permutation(gv):
        if len(mis) == n_mis:
            break
        peers_given = {puz[p] for p in PEERS[i] if puz[p] and p not in mis}
        wrong = [d for d in peers_given if d != puz[i]]
        if not wrong:
            continue
        w = int(rng.choice(wrong))
        p = np.full(9, (1 - read_p - true_p) / 7); p[w - 1] = read_p; p[puz[i] - 1] = true_p
        probs[i] = p; grid0[i] = w; mis.append(int(i))
    return grid0, probs, puz, sol, mis
