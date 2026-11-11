import numpy as np
from p7.integration import correct_and_solve, read_mosaic
from p7.data import is_blank
from p7.csp import solve
from tests.helpers import synthetic_case, KNOWN, KNOWN_SOL, digits


def confident_probs(grid, conf=0.99):
    p = np.full((81, 9), 1 / 9)
    for i, v in enumerate(grid):
        if v:
            q = np.full(9, (1 - conf) / 8); q[v - 1] = conf; p[i] = q
    return p


# ---------- normal ----------
def test_end_to_end_with_misreads():
    for seed, k in [(1, 1), (2, 2), (3, 3)]:
        grid0, probs, puz, sol, mis = synthetic_case(seed, k)
        o = correct_and_solve(grid0, probs, algo="astar", goal="unique")
        assert o.status == "solved" and o.solution == sol
        assert {c["cell"] for c in o.corrections} == set(mis)
        assert o.ac3_conflict and o.initial_conflicts                # AC-3 spotted it first
        assert all(c["cost"] >= 0 for c in o.corrections)


def test_all_three_algorithms_solve_and_modes_ordering():
    grid0, probs, puz, sol, mis = synthetic_case(4, 2)
    for algo in ("ucs", "greedy", "astar"):
        assert correct_and_solve(grid0, probs, algo=algo, goal="unique").solution == sol
    assert correct_and_solve(grid0, probs, mode="none").status == "unreadable"   # no correction -> fails


def test_blank_detection_rule():
    assert is_blank(np.random.default_rng(0).random((28, 28)) * 0.03)
    assert not is_blank(np.eye(28))


class FakeModel:
    def probs(self, x, T=None):
        p = np.full((len(x), 9), 0.01); p[:, 4] = 0.92
        return p


def test_read_mosaic_uses_brightness_for_blanks():
    mosaic = np.zeros((252, 252), dtype=np.float32)
    mosaic[:28, :28] = 1.0                                            # one bright cell
    rd = read_mosaic(FakeModel(), mosaic)
    assert rd.grid[0, 0] == 5 and (rd.grid.reshape(-1)[1:] == 0).all()


# ---------- edge ----------
def test_edge_clean_reading_needs_no_correction():
    o = correct_and_solve(digits(KNOWN), confident_probs(digits(KNOWN)))
    assert o.status == "solved" and o.corrections == [] and o.nodes_expanded == 0


def test_edge_uniqueness_goal_rejects_ambiguous_grid():
    g = [0] * 81
    g[0] = 1                                                          # millions of solutions
    o = correct_and_solve(g, confident_probs(g), goal="unique")
    assert o.status == "unreadable"
    assert correct_and_solve(g, confident_probs(g), goal="solvable").status == "solved"


# ---------- impossible ----------
def test_impossible_confident_duplicate_gives_clear_unreadable_message():
    g = digits(KNOWN)
    g[2] = 5                                                          # second 5 in row 1 (was blank)
    p = confident_probs(g, conf=0.99999999)                           # no alternative above the floor
    o = correct_and_solve(g, p)
    assert o.status == "unreadable" and o.solution is None
    assert "UNREADABLE" in o.message and "Least confident" in o.message and len(o.least_confident) > 0


def test_impossible_too_many_misreads():
    grid0, probs, *_ = synthetic_case(7, 4)
    o = correct_and_solve(grid0, probs, goal="unique", max_corr=3)
    assert o.status == "unreadable" and "UNREADABLE" in o.message
