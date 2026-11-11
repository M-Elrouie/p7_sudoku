import pytest
from p7.csp import solve, find_conflicts, ac3_detects_conflict, is_valid_solution
from p7.generator import make_puzzle
import numpy as np
from tests.helpers import KNOWN, KNOWN_SOL, digits

VARIANTS = [dict(fc=False), dict(fc=True), dict(fc=False, ac3_on=True), dict(fc=True, ac3_on=True, order="mrv")]


# ---------------- normal (5) ----------------
@pytest.mark.parametrize("kw", VARIANTS)
def test_known_puzzle_every_variant(kw):                    # normal 1-4
    r = solve(digits(KNOWN), **kw)
    assert r.solvable and r.solutions[0] == digits(KNOWN_SOL)


def test_variants_agree_on_random_puzzles():                # normal 5
    rng = np.random.default_rng(3)
    for _ in range(4):
        puz, sol = make_puzzle(rng, 28)
        for kw in VARIANTS:
            assert solve(puz, **kw).solutions[0] == sol


def test_fc_prunes_more_than_plain_bt():                    # normal 6 (counters)
    puz, _ = make_puzzle(np.random.default_rng(5), 26)
    bt, fc, ac = (solve(puz, fc=False), solve(puz, fc=True), solve(puz, fc=True, ac3_on=True))
    assert bt.stats.assignments > fc.stats.assignments > ac.stats.assignments
    assert ac.stats.revisions > 0 and fc.stats.prunings > 0


# ---------------- edge (3) ----------------
def test_edge_already_complete_grid():
    r = solve(digits(KNOWN_SOL), fc=True, ac3_on=True)
    assert r.solutions == [digits(KNOWN_SOL)] and r.stats.assignments == 0


def test_edge_empty_grid_is_solvable():
    for kw in ({"fc": True}, {"fc": True, "ac3_on": True, "order": "mrv"}):
        r = solve([0] * 81, **kw)
        assert r.solvable and is_valid_solution(r.solutions[0], [0] * 81)


def test_edge_multiple_solutions_counted_and_cap_aborts():
    r = solve([0] * 81, fc=True, ac3_on=True, max_solutions=2)
    assert len(r.solutions) == 2                                   # empty grid has many
    capped = solve([0] * 81, fc=False, max_assignments=10)  # tiny cap -> aborted, not crash
    assert capped.stats.aborted and not capped.solvable


# ---------------- impossible (2) ----------------
@pytest.mark.parametrize("kw", VARIANTS[:3])
def test_impossible_duplicate_givens(kw):
    g = [0] * 81
    g[0] = g[8] = 5                      # two 5s in row 1
    assert find_conflicts(g) and ac3_detects_conflict(g)
    r = solve(g, **kw)
    assert not r.solvable and "No solution" in r.message and r.stats.root_conflict


def test_impossible_valid_givens_but_unsolvable():
    g = [0] * 81
    g[0:8] = [1, 2, 3, 4, 5, 6, 7, 8]    # row 1 needs a 9 in the last cell ...
    g[17] = 9                            # ... but column 9 already holds a 9
    assert not find_conflicts(g)
    for kw in VARIANTS[:3]:
        r = solve(g, **kw)
        assert not r.solvable and "No solution" in r.message
