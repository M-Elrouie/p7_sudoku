"""Puzzle generator (seeded) and 9x9 handwritten-digit mosaic builder with injected hard digits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .csp import solve


def random_full_grid(rng: np.random.Generator) -> List[int]:
    """Complete valid grid by randomised backtracking."""
    g = [0] * 81
    row = [0] * 9; col = [0] * 9; box = [0] * 9

    def rec(i):
        if i == 81:
            return True
        r, c = divmod(i, 9)
        b = (r // 3) * 3 + c // 3
        for v in rng.permutation(9) + 1:
            bit = 1 << int(v)
            if not (row[r] & bit or col[c] & bit or box[b] & bit):
                g[i] = int(v); row[r] |= bit; col[c] |= bit; box[b] |= bit
                if rec(i + 1):
                    return True
                row[r] &= ~bit; col[c] &= ~bit; box[b] &= ~bit
        g[i] = 0
        return False

    rec(0)
    return g


def make_puzzle(rng: np.random.Generator, givens: int, max_tries: int = 30):
    """Remove cells (random order) while the solution stays unique, until `givens` remain.
    Returns (puzzle, solution) as flat int lists. Retries with a new grid if stuck."""
    for _ in range(max_tries):
        sol = random_full_grid(rng)
        puz = list(sol)
        n = 81
        for c in rng.permutation(81):
            if n <= givens:
                break
            keep = puz[c]
            puz[c] = 0
            if len(solve(puz, fc=True, ac3_on=True, max_solutions=2).solutions) == 1:
                n -= 1
            else:
                puz[c] = keep
        if n == givens:
            return puz, sol
    raise RuntimeError(f"could not build a unique puzzle with {givens} givens")


class ImagePools:
    """Easy / hard test images per digit.

    easy[d]: test images of digit d that the network classifies correctly.
    hard[d]: images of digit d the network gets WRONG, but with the true digit among its top 3
             (so a correction is possible). If fewer than 3 exist (network too accurate), the
             images with the lowest p(true digit) are added. Documented injection rule.
    """

    def __init__(self, model, x_test: np.ndarray, y_test: np.ndarray):
        self.x = x_test
        p = model.probs(x_test)
        pred = p.argmax(1)
        p_true = p[np.arange(len(y_test)), y_test]
        rank = (p > p_true[:, None]).sum(1)
        self.easy: Dict[int, np.ndarray] = {}
        self.hard: Dict[int, np.ndarray] = {}
        self.hard_is_misread: Dict[int, bool] = {}
        for d in range(9):
            of_d = np.where(y_test == d)[0]
            self.easy[d] = of_d[pred[of_d] == d]
            h = of_d[(pred[of_d] != d) & (rank[of_d] <= 2)]
            mis = True
            if len(h) < 3:
                extra = of_d[np.argsort(p_true[of_d])][:5]
                h = np.unique(np.concatenate([h, extra]))
                mis = bool((pred[h] != d).any())
            self.hard[d] = h
            self.hard_is_misread[d + 1] = mis


@dataclass
class Instance:
    mosaic: np.ndarray          # (252,252) floats in [0,1]
    puzzle: List[int]           # the true puzzle
    solution: List[int]         # its unique solution
    injected: List[int]         # cells that got a hard image


def build_instance(rng: np.random.Generator, pools: ImagePools, puzzle: List[int],
                   solution: List[int], n_misreads: int, noise: float = 0.03) -> Instance:
    givens = [i for i in range(81) if puzzle[i]]
    inj = sorted(int(c) for c in rng.choice(givens, size=min(n_misreads, len(givens)), replace=False)) \
        if n_misreads else []
    mosaic = np.zeros((252, 252), dtype=np.float32)
    for i in range(81):
        r, c = divmod(i, 9)
        if puzzle[i]:
            d = puzzle[i] - 1
            pool = pools.hard[d] if i in inj else pools.easy[d]
            cell = pools.x[int(rng.choice(pool))].reshape(28, 28)
        else:
            cell = rng.random((28, 28)).astype(np.float32) * noise   # blank paper noise
        mosaic[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = cell
    return Instance(mosaic, list(puzzle), list(solution), inj)
