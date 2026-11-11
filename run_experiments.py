"""python run_experiments.py [--quick]   -> tables (csv + md) and plots in results/"""
import argparse
import numpy as np
from p7.data import load_digits
from p7.nn import SoftmaxNet
from p7.generator import ImagePools
from p7.eval import run_all

ap = argparse.ArgumentParser()
ap.add_argument("--exp", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--seeds", type=int, default=5, help="number of seeds (>=5 required by the checklist)")
ap.add_argument("--puzzles", type=int, default=100, help="puzzles per misread level (exp 1 and 3)")
ap.add_argument("--puzzles2", type=int, default=20, help="puzzles per givens level (exp 2)")
ap.add_argument("--quick", action="store_true", help="smoke run: 2 seeds, 10 / 5 puzzles")
ap.add_argument("--goal", choices=["solvable", "unique"], default="solvable",
                help="search goal: puzzle solvable (minimum version) or exactly one solution (extension)")
ap.add_argument("--out", default="results")
a = ap.parse_args()
if a.quick:
    a.seeds, a.puzzles, a.puzzles2 = 2, 10, 5

d = load_digits()
model = SoftmaxNet.load("data/mlp.npz")
pools = ImagePools(model, d["x_test"], d["y_test"])
print("data:", d["source"], "| seeds:", a.seeds, "| puzzles/level:", a.puzzles, a.puzzles2)
run_all(set(a.exp), model, pools, a.out, list(range(a.seeds)), a.puzzles, a.puzzles2, a.goal)
print("done ->", a.out)
