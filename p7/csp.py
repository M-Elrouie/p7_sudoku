"""Sudoku as a CSP: Backtracking, Forward Checking and AC-3 as switchable parts.

Variables : 81 cells (index = 9*row + col).
Domains   : bitmasks, bit d set  <=>  digit d is still possible (d = 1..9).
Constraint: any two cells in the same row / column / box differ (binary !=).

Switches of `solve`:
    fc=False, ac3=False  -> plain backtracking (consistency check on assignment only)
    fc=True               -> + Forward Checking (remove the assigned digit from peers)
    ac3=True              -> + AC-3 at the root and re-run after every assignment
                             (it subsumes FC, and also propagates chains of singletons)
Counters (CSPStats): assignments, backtracks, revisions (AC-3 revise calls), prunings.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

ALL = 0b1111111110  # bits 1..9


def _build_units():
    units = [[r * 9 + c for c in range(9)] for r in range(9)]
    units += [[r * 9 + c for r in range(9)] for c in range(9)]
    for br in range(3):
        for bc in range(3):
            units.append([(br * 3 + i) * 9 + bc * 3 + j for i in range(3) for j in range(3)])
    peers = [set() for _ in range(81)]
    for u in units:
        for a in u:
            for b in u:
                if a != b:
                    peers[a].add(b)
    return units, [sorted(p) for p in peers]


UNITS, PEERS = _build_units()


@dataclass
class CSPStats:
    assignments: int = 0
    backtracks: int = 0
    revisions: int = 0
    prunings: int = 0
    aborted: bool = False
    root_conflict: bool = False

    def add(self, other: "CSPStats") -> None:
        self.assignments += other.assignments
        self.backtracks += other.backtracks
        self.revisions += other.revisions
        self.prunings += other.prunings
        self.aborted |= other.aborted
        self.root_conflict |= other.root_conflict


@dataclass
class CSPResult:
    solutions: List[List[int]] = field(default_factory=list)
    stats: CSPStats = field(default_factory=CSPStats)

    @property
    def solvable(self) -> bool:
        return len(self.solutions) > 0

    @property
    def message(self) -> str:
        if self.solvable:
            return f"{len(self.solutions)} solution(s) found"
        if self.stats.aborted:
            return "search aborted (assignment cap reached): undecided"
        if self.stats.root_conflict:
            return "No solution: the givens already conflict (two equal digits in one row/column/box)"
        return "No solution: the puzzle is inconsistent"


class _Abort(Exception):
    pass


def _flat(grid) -> List[int]:
    return [int(x) for x in np.asarray(grid).reshape(-1)]


def find_conflicts(grid) -> List[dict]:
    """Duplicate-digit groups: one entry per (unit, digit) that occurs >= 2 times."""
    g = _flat(grid)
    out = []
    for ui, unit in enumerate(UNITS):
        by_digit = {}
        for c in unit:
            if g[c]:
                by_digit.setdefault(g[c], []).append(c)
        for d, cells in by_digit.items():
            if len(cells) >= 2:
                out.append({"unit": ui, "digit": d, "cells": cells})
    return out


def ac3(dom: List[int], queue, stats: CSPStats) -> bool:
    """AC-3 for binary '!=' constraints. Returns False if some domain becomes empty."""
    q = deque(queue)
    while q:
        xi, xj = q.popleft()
        stats.revisions += 1
        dj = dom[xj]
        if dj & (dj - 1) == 0 and dom[xi] & dj:  # xj is a singleton that is still in D(xi)
            dom[xi] &= ~dj
            stats.prunings += 1
            if dom[xi] == 0:
                return False
            for xk in PEERS[xi]:
                if xk != xj:
                    q.append((xk, xi))
    return True


def _all_arcs():
    return [(xi, xj) for xi in range(81) for xj in PEERS[xi]]


def ac3_detects_conflict(grid) -> bool:
    """Root AC-3 only (no search): True if it empties a domain, e.g. two equal givens in a row."""
    g = _flat(grid)
    dom = [(1 << v) if v else ALL for v in g]
    return not ac3(dom, _all_arcs(), CSPStats())


def solve(grid, fc: bool = True, ac3_on: bool = False, max_solutions: int = 1,
          max_assignments: Optional[int] = None, order: str = "static") -> CSPResult:
    """Solve a Sudoku (0 = blank). `order` is 'static' (row-major) or 'mrv'."""
    g = _flat(grid)
    stats = CSPStats()
    sols: List[List[int]] = []
    cap = max_assignments

    def count_assign():
        stats.assignments += 1
        if cap is not None and stats.assignments > cap:
            raise _Abort()

    try:
        if not fc and not ac3_on:
            _plain_bt(g, stats, sols, max_solutions, count_assign)
        else:
            _domain_search(g, stats, sols, max_solutions, count_assign, fc, ac3_on, order)
    except _Abort:
        stats.aborted = True
    return CSPResult(sols, stats)


def _plain_bt(a, stats, sols, max_solutions, count_assign):
    for i, v in enumerate(a):
        if v and any(a[p] == v for p in PEERS[i]):
            stats.root_conflict = True
            return
    empties = [i for i in range(81) if a[i] == 0]

    def rec(k):
        if k == len(empties):
            sols.append(list(a))
            return len(sols) >= max_solutions
        i = empties[k]
        for v in range(1, 10):
            if all(a[p] != v for p in PEERS[i]):
                a[i] = v
                count_assign()
                if rec(k + 1):
                    return True
                a[i] = 0
                stats.backtracks += 1
        return False

    rec(0)


def _domain_search(g, stats, sols, max_solutions, count_assign, fc, use_ac3, order):
    dom = [(1 << v) if v else ALL for v in g]
    assigned = [bool(v) for v in g]
    # root inference
    if use_ac3:
        if not ac3(dom, _all_arcs(), stats):
            stats.root_conflict = True
            return
    else:  # forward checking from the givens
        for i, v in enumerate(g):
            if v:
                b = 1 << v
                for p in PEERS[i]:
                    if dom[p] & b:
                        dom[p] &= ~b
                        stats.prunings += 1
                        if dom[p] == 0:
                            stats.root_conflict = True
                            return

    def pick(dom):
        best, best_n = -1, 99
        for i in range(81):
            if use_ac3:
                if dom[i] & (dom[i] - 1) == 0:
                    continue  # singleton: already decided by propagation
            elif assigned[i]:
                continue
            if order == "static":
                return i
            n = dom[i].bit_count()
            if n < best_n:
                best, best_n = i, n
        return best

    def rec(dom):
        x = pick(dom)
        if x == -1:
            sols.append([d.bit_length() - 1 for d in dom])
            return len(sols) >= max_solutions
        d = dom[x]
        for v in range(1, 10):
            b = 1 << v
            if not d & b:
                continue
            new = list(dom)
            new[x] = b
            count_assign()
            ok = True
            if use_ac3:
                ok = ac3(new, [(p, x) for p in PEERS[x]], stats)
            else:
                for p in PEERS[x]:
                    if new[p] & b:
                        new[p] &= ~b
                        stats.prunings += 1
                        if new[p] == 0:
                            ok = False
                            break
            if ok:
                assigned[x] = True
                done = rec(new)
                assigned[x] = False
                if done:
                    return True
            stats.backtracks += 1
        return False

    rec(dom)


def is_valid_solution(sol: Sequence[int], puzzle: Sequence[int]) -> bool:
    s, p = _flat(sol), _flat(puzzle)
    if any(p[i] and p[i] != s[i] for i in range(81)):
        return False
    return all(sorted(s[c] for c in u) == list(range(1, 10)) for u in UNITS)
