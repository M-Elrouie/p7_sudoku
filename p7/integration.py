"""NN reads the grid -> AC-3 finds a conflict -> Search tries cheapest corrections -> CSP tests each.

Modes:
  'nn'     : correction search guided by the network's probabilities (ucs / greedy / astar)
  'none'   : no correction at all (just try to solve the first reading)
  'random' : random correction sets (random cells, random digits), same max size, fixed budget
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .csp import CSPStats, ac3_detects_conflict, find_conflicts, solve
from .data import is_blank
from .search import CorrectionProblem, search


@dataclass
class Reading:
    grid: np.ndarray       # (9,9) ints, 0 = blank
    probs: np.ndarray      # (81,9); blank cells = uniform 1/9 (never used)


def read_mosaic(model, mosaic: np.ndarray, T: Optional[float] = None) -> Reading:
    cells = mosaic.reshape(9, 28, 9, 28).transpose(0, 2, 1, 3).reshape(81, 28, 28)
    blank = np.array([is_blank(c) for c in cells])
    probs = np.full((81, 9), 1 / 9)
    if (~blank).any():
        probs[~blank] = model.probs(cells[~blank].reshape(-1, 784), T)
    grid = np.where(blank, 0, probs.argmax(1) + 1).reshape(9, 9)
    return Reading(grid, probs)


@dataclass
class Outcome:
    status: str                         # 'solved' | 'unreadable'
    message: str
    solution: Optional[List[int]] = None
    corrections: List[dict] = field(default_factory=list)   # cell,row,col,read,new,cost
    initial_conflicts: list = field(default_factory=list)
    ac3_conflict: bool = False
    nodes_expanded: int = 0
    nodes_generated: int = 0
    goal_checks: int = 0
    csp_calls: int = 0
    csp: CSPStats = field(default_factory=CSPStats)
    least_confident: list = field(default_factory=list)
    trace: list = field(default_factory=list)


def least_confident_cells(grid0, probs, k: int = 5):
    g = np.asarray(grid0).reshape(-1)
    cells = [i for i in range(81) if g[i]]
    cells.sort(key=lambda i: probs[i].max())
    return [dict(cell=i, row=i // 9, col=i % 9, read=int(g[i]), confidence=float(probs[i].max()))
            for i in cells[:k]]


def make_goal(goal: str, totals: CSPStats, counter: dict, fc=True, ac3_on=True, cap=200000):
    def goal_fn(grid):
        g = grid.grid if hasattr(grid, "grid") else grid
        if find_conflicts(g):          # duplicate givens = what root AC-3 detects; cheap pre-check
            return False
        r = solve(g, fc=fc, ac3_on=ac3_on, max_solutions=2 if goal == "unique" else 1,
                  max_assignments=cap)
        counter["calls"] += 1
        totals.add(r.stats)
        counter["last"] = r
        return len(r.solutions) == 1 if goal == "unique" else r.solvable
    return goal_fn


def correct_and_solve(grid0, probs, mode: str = "nn", algo: str = "astar", goal: str = "solvable",
                      max_corr: int = 3, n_cand: int = 12, n_alt: int = 2, random_budget: int = 300,
                      rng: Optional[np.random.Generator] = None, fc: bool = True, ac3_on: bool = True
                      ) -> Outcome:
    grid0 = np.asarray(grid0).reshape(-1)
    probs = np.asarray(probs).reshape(81, 9)
    out = Outcome("unreadable", "")
    out.initial_conflicts = find_conflicts(grid0)
    out.ac3_conflict = ac3_detects_conflict(grid0)
    counter = {"calls": 0, "last": None}
    goal_fn = make_goal(goal, out.csp, counter, fc, ac3_on)

    def finish(found_grid, corrs):
        sol = solve(found_grid, fc=fc, ac3_on=ac3_on).solutions[0]
        out.status, out.solution = "solved", sol
        out.corrections = [dict(cell=c, row=c // 9, col=c % 9, read=int(grid0[c]), new=d, cost=float(cost))
                           for c, d, cost in corrs]
        out.message = ("Solved with no corrections" if not corrs else
                       f"Solved after {len(corrs)} correction(s), total cost {sum(c[2] for c in corrs):.3f}")
        out.csp_calls = counter["calls"]
        return out

    if goal_fn(tuple(grid0)):                                   # first reading already fine
        return finish(tuple(grid0), [])
    if mode == "none":
        pass
    elif mode == "nn":
        prob = CorrectionProblem(grid0, probs, max_corr, n_cand, n_alt)
        res = search(prob, goal_fn, algo)
        out.nodes_expanded, out.nodes_generated, out.goal_checks = res.expanded, res.generated, res.goal_checks
        out.trace = res.trace
        if res.found is not None:
            return finish(res.found.grid, res.found.corrections)
    elif mode == "random":
        rng = rng or np.random.default_rng(0)
        givens = [i for i in range(81) if grid0[i]]
        for _ in range(random_budget):
            k = int(rng.integers(1, max_corr + 1))
            cells = rng.choice(givens, size=min(k, len(givens)), replace=False)
            g, corrs = list(grid0), []
            for c in cells:
                new = int(rng.choice([d for d in range(1, 10) if d != grid0[c]]))
                g[c] = new
                corrs.append((int(c), new, 0.0))
            out.goal_checks += 1
            if goal_fn(tuple(g)):
                return finish(tuple(g), corrs)
    out.csp_calls = counter["calls"]
    out.least_confident = least_confident_cells(grid0, probs)
    cells = ", ".join(f"(r{c['row'] + 1},c{c['col'] + 1}) read {c['read']} @ {c['confidence']:.2f}"
                      for c in out.least_confident)
    out.message = (f"UNREADABLE: no solvable reading found within {max_corr} corrections. "
                   f"Least confident cells: {cells}")
    return out
