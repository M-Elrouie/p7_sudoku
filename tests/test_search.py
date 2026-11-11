import math
import numpy as np
from p7.csp import solve, find_conflicts
from p7.search import CorrectionProblem, search
from p7.integration import make_goal
from p7.csp import CSPStats
from tests.helpers import synthetic_case


def run(case, algo, goal="solvable", **kw):
    grid0, probs, *_ = case
    prob = CorrectionProblem(grid0, probs, **kw)
    cnt = {"calls": 0}
    res = search(prob, make_goal(goal, CSPStats(), cnt), algo)
    return prob, res


def test_spec_example_seven_read_as_one():
    """Known answer: the net reads a 7 as a 1 (0.52 vs 0.45) -> ONE correction, cost ln(.52/.45)."""
    case = synthetic_case(11, 1)
    grid0, probs, puz, sol, mis = case
    for algo in ("ucs", "greedy", "astar"):
        prob, res = run(case, algo, goal="unique")
        assert res.found is not None
        (cell, new, cost), = res.found.corrections
        assert cell == mis[0] and new == puz[cell]
        assert math.isclose(cost, math.log(0.52 / 0.45), rel_tol=1e-9)
        assert res.expanded >= 0 and res.generated >= 1 and res.goal_checks >= 1   # counters


def test_astar_matches_ucs_cost_and_expands_no_more():
    for seed in range(6):
        case = synthetic_case(seed, 2)
        _, u = run(case, "ucs", goal="unique")
        _, a = run(case, "astar", goal="unique")
        assert u.found and a.found
        assert math.isclose(u.cost, a.cost, rel_tol=1e-9)           # A* optimal (h admissible)
        assert a.expanded <= u.expanded


def test_heuristic_is_admissible_on_root():
    for seed in range(8):
        case = synthetic_case(100 + seed, 2)
        prob, u = run(case, "ucs", goal="unique")
        assert prob.heuristic(prob.root()) <= u.cost + 1e-9


def test_costs_never_negative_and_limit_respected():
    case = synthetic_case(5, 3)
    prob, res = run(case, "ucs", goal="unique", max_corr=3)
    assert all(c >= 0 for _, _, c in res.found.corrections) and len(res.found.corrections) <= 3


def test_recovers_exact_solution_three_misreads():
    for seed in range(4):
        case = synthetic_case(200 + seed, 3)
        _, res = run(case, "astar", goal="unique")
        assert res.found is not None
        assert solve(res.found.grid).solutions[0] == case[3]


def test_search_gives_up_beyond_limit():
    case = synthetic_case(7, 4)                                    # needs 4 corrections, max is 3
    _, res = run(case, "astar", goal="unique", max_corr=3)
    assert res.found is None
