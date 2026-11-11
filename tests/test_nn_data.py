import numpy as np
from p7.nn import SoftmaxNet, accuracy, ece, fit_temperature, softmax, confusion_matrix
from p7.generator import make_puzzle, random_full_grid
from p7.csp import solve


def toy(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 9, n)
    centers = np.random.default_rng(123).normal(0, 1.0, (9, 20))   # same classes for every split
    x = centers[y] + rng.normal(0, 1.0, (n, 20))
    return x.astype(np.float32), y


def test_mlp_learns_and_beats_or_matches_baseline():
    x, y = toy(); xt, yt = toy(600, 1)
    mlp = SoftmaxNet([20, 32, 9], 0).fit(x, y, epochs=25, lr=3e-3, verbose=False)
    lr = SoftmaxNet([20, 9], 0).fit(x, y, epochs=25, lr=3e-3, verbose=False)
    a_m, a_l = accuracy(mlp.probs(xt), yt), accuracy(lr.probs(xt), yt)
    assert a_m > 0.6 and a_m >= a_l - 0.03


def test_ece_and_temperature_scaling():
    assert ece(np.eye(9)[[0, 1, 2]], np.array([0, 1, 2])) < 1e-9
    rng = np.random.default_rng(0)
    y = rng.integers(0, 9, 3000)
    logits = rng.normal(0, 1, (3000, 9)); logits[np.arange(3000), y] += 2.0
    over = logits * 4                                                  # over-confident
    T = fit_temperature(over, y)
    assert T > 1.5 and ece(softmax(over / T), y) < ece(softmax(over), y)


def test_confusion_matrix_counts():
    cm = confusion_matrix(np.array([0, 1, 1]), np.array([0, 1, 2]))
    assert cm.sum() == 3 and cm[2, 1] == 1


def test_generator_unique_reproducible():
    p1, s1 = make_puzzle(np.random.default_rng(9), 30)
    p2, s2 = make_puzzle(np.random.default_rng(9), 30)
    assert p1 == p2 and s1 == s2                                       # seeded
    assert sum(v != 0 for v in p1) == 30
    assert len(solve(p1, fc=True, ac3_on=True, max_solutions=2).solutions) == 1


def test_random_full_grid_valid():
    from p7.csp import is_valid_solution
    g = random_full_grid(np.random.default_rng(1))
    assert is_valid_solution(g, [0] * 81)
