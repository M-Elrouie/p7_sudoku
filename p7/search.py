"""Search over *correction sets* of the network's first reading.

State  : the grid after a set of corrections (each cell corrected at most once).
Action : change one uncertain cell to one of its top alternative digits.
Cost   : ln p(read) - ln p(new)  (>= 0 because the read digit is the arg-max).
Goal   : supplied by the caller (the CSP says "solvable", or "exactly one solution").
Limit  : at most `max_corr` corrections (default 3).

Algorithms (same code, different priority):  UCS: g   |  Greedy: h   |  A*: g + h.

Duplicate-free expansion: a node only corrects cells with an index larger than its last
corrected cell, so every correction *set* is generated exactly once (tree search, no closed
list needed, A* with an admissible h stays optimal).

Heuristic (admissible): a (unit, digit) group with k equal digits needs >= k-1 changed
cells. Take a greedy packing of cell-disjoint groups; in each take the (k-1) cheapest
changeable cells' cheapest alternative cost; add up. If the groups need more changes than
corrections left (or cells that cannot change) the node is a dead end (h = inf).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .csp import find_conflicts

INF = float("inf")


@dataclass
class Node:
    grid: Tuple[int, ...]
    corrections: Tuple[Tuple[int, int, float], ...] = ()   # (cell, new_digit, cost)
    g: float = 0.0
    last: int = -1


@dataclass
class SearchResult:
    found: Optional[Node]
    expanded: int = 0
    generated: int = 1
    goal_checks: int = 0
    trace: List[Tuple[int, float, float, tuple]] = field(default_factory=list)  # (order,g,h,corrs)
    exhausted_cap: bool = False

    @property
    def cost(self) -> float:
        return self.found.g if self.found else INF


class CorrectionProblem:
    def __init__(self, grid0, probs, max_corr: int = 3, n_cand: int = 12, n_alt: int = 2,
                 floor: float = 1e-6):
        self.grid0 = tuple(int(x) for x in np.asarray(grid0).reshape(-1))
        self.probs = np.asarray(probs, dtype=float).reshape(81, 9)
        self.max_corr = max_corr
        conf = self.probs.max(axis=1)
        givens = [i for i in range(81) if self.grid0[i]]
        self.alts = {}
        for c in givens:
            read = self.grid0[c]
            order = np.argsort(-self.probs[c])
            alt = []
            for j in order:
                d = int(j) + 1
                if d != read and self.probs[c, j] >= floor and len(alt) < n_alt:
                    alt.append((d, self._cost(c, read, d)))
            if alt:
                self.alts[c] = alt
        ranked = sorted(self.alts, key=lambda c: conf[c])[:n_cand]   # least confident first
        self.cand_cells = sorted(ranked)
        self.cand_set = set(self.cand_cells)
        self.mc = {c: self.alts[c][0][1] for c in self.cand_cells}  # cheapest change per cell

    def _cost(self, cell, read, new) -> float:
        p = self.probs[cell]
        return max(0.0, math.log(max(p[read - 1], 1e-12)) - math.log(max(p[new - 1], 1e-12)))

    def root(self) -> Node:
        return Node(grid=self.grid0)

    def children(self, node: Node):
        if len(node.corrections) >= self.max_corr:
            return
        for c in self.cand_cells:
            if c <= node.last:
                continue
            for d, cost in self.alts[c]:
                g = list(node.grid)
                g[c] = d
                yield Node(tuple(g), node.corrections + ((c, d, cost),), node.g + cost, c)

    def heuristic(self, node: Node) -> float:
        groups = find_conflicts(node.grid)
        if not groups:
            return 0.0
        remaining = self.max_corr - len(node.corrections)
        info = []
        for gr in groups:
            cells = gr["cells"]
            need = len(cells) - 1
            costs = sorted(self.mc[c] for c in cells if c in self.cand_set and c > node.last)
            if len(costs) < need:
                return INF
            info.append((sum(costs[:need]), need, set(cells)))
        info.sort(key=lambda t: -t[0])
        used, h, changes = set(), 0.0, 0
        for lb, need, cells in info:
            if used.isdisjoint(cells):
                used |= cells
                h += lb
                changes += need
        return INF if changes > remaining else h


def search(problem: CorrectionProblem, goal: Callable[[Node], bool], algo: str = "ucs",
           max_expansions: int = 20000) -> SearchResult:
    """algo in {'ucs','greedy','astar'}. Goal test happens when a node is popped."""
    res = SearchResult(found=None)
    use_h = algo in ("greedy", "astar")
    root = problem.root()
    h0 = problem.heuristic(root) if use_h else 0.0
    if h0 == INF:
        return res
    counter = 0
    key = {"ucs": lambda g, h: (g, 0.0), "greedy": lambda g, h: (h, g), "astar": lambda g, h: (g + h, h)}[algo]
    heap = [(key(0.0, h0), counter, root, h0)]
    while heap:
        _, _, node, h = heapq.heappop(heap)
        res.goal_checks += 1
        res.trace.append((res.goal_checks, node.g, h, node.corrections))
        if goal(node):
            res.found = node
            return res
        res.expanded += 1
        if res.expanded >= max_expansions:
            res.exhausted_cap = True
            return res
        for ch in problem.children(node):
            hc = problem.heuristic(ch) if use_h else 0.0
            if hc == INF:
                continue
            counter += 1
            res.generated += 1
            heapq.heappush(heap, (key(ch.g, hc), counter, ch, hc))
    return res
