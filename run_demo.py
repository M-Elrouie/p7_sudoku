"""python run_demo.py [--misreads 2] [--seed 0] [--givens 30] [--impossible]

Shows: the input (mosaic + first reading), AC-3 conflict detection, the search trace (UCS /
Greedy / A*), the correction log, the CSP with and without FC / AC-3, and the system with the
NN-guided correction ON vs OFF. Plots are written to results/demo_*.png.
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from p7.data import load_digits
from p7.nn import SoftmaxNet
from p7.generator import ImagePools, build_instance, make_puzzle
from p7.integration import correct_and_solve, read_mosaic
from p7.csp import solve, find_conflicts

ap = argparse.ArgumentParser()
ap.add_argument("--misreads", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--givens", type=int, default=30)
ap.add_argument("--goal", choices=["solvable", "unique"], default="unique")
ap.add_argument("--impossible", action="store_true", help="inject 5 misreads (> 3 allowed) -> 'unreadable'")
a = ap.parse_args()
if a.impossible:
    a.misreads = 5
os.makedirs("results", exist_ok=True)


def show(grid, title="", marks=()):
    g = np.asarray(grid).reshape(9, 9)
    lines = []
    for r in range(9):
        if r % 3 == 0 and r:
            lines.append("  ------+-------+------")
        row = ""
        for c in range(9):
            if c % 3 == 0 and c:
                row += "| "
            row += ("." if g[r, c] == 0 else str(g[r, c])) + ("*" if (r * 9 + c) in marks else " ")
        lines.append("  " + row)
    print(title); print("\n".join(lines)); print()


def draw_grid(ax, grid, title, red=(), green=(), given=None):
    ax.set_xlim(0, 9); ax.set_ylim(9, 0); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=10)
    for r in range(9):
        for c in range(9):
            i = r * 9 + c
            v = np.asarray(grid).reshape(-1)[i]
            if i in red: ax.add_patch(Rectangle((c, r), 1, 1, color="#f6b3b3"))
            if i in green: ax.add_patch(Rectangle((c, r), 1, 1, color="#b9e6b9"))
            if v:
                bold = given is not None and given[i]
                ax.text(c + .5, r + .55, str(v), ha="center", va="center", fontsize=13,
                        fontweight="bold" if bold else "normal", color="black" if bold or given is None else "#335")
    for k in range(10):
        ax.plot([0, 9], [k, k], color="k", lw=2 if k % 3 == 0 else .5)
        ax.plot([k, k], [0, 9], color="k", lw=2 if k % 3 == 0 else .5)


# ----------------------------------------------------------------- build the instance
d = load_digits()
model = SoftmaxNet.load("data/mlp.npz")
pools = ImagePools(model, d["x_test"], d["y_test"])
print(f"digit data source: {d['source']}\n")
rng = np.random.default_rng(a.seed)
puz, sol = make_puzzle(rng, a.givens)
for attempt in range(300):                    # demo only: re-draw images until the NN really misreads k cells
    inst = build_instance(np.random.default_rng([a.seed, attempt]), pools, puz, sol, a.misreads)
    rd = read_mosaic(model, inst.mosaic)
    wrong = [i for i in range(81) if rd.grid.reshape(-1)[i] != puz[i]]
    if len(wrong) == a.misreads:
        break
print(f"puzzle: {a.givens} givens (unique solution) | network misreads: {len(wrong)} cells {wrong}\n")
show(puz, "TRUE puzzle:")
show(rd.grid, "NN first reading (* = misread):", set(wrong))
conf = find_conflicts(rd.grid)
out = correct_and_solve(rd.grid, rd.probs, algo="astar", goal=a.goal)
print(f"AC-3 on the first reading: conflict = {out.ac3_conflict}  ({len(conf)} duplicate groups: "
      + ", ".join(f"digit {c['digit']} in cells {c['cells']}" for c in conf[:4]) + ")")

# ----------------------------------------------------------------- search trace + algorithms
print("\nSEARCH (UCS / Greedy / A*): nodes expanded / generated / total correction cost")
res = {}
for algo in ("ucs", "greedy", "astar"):
    res[algo] = correct_and_solve(rd.grid, rd.probs, algo=algo, goal=a.goal)
    o = res[algo]
    print(f"  {algo:6s} expanded={o.nodes_expanded:5d} generated={o.nodes_generated:6d} "
          f"cost={sum(c['cost'] for c in o.corrections):.3f} status={o.status}")
print("\nA* trace (first 8 popped nodes: order, g, h, corrections [cell->digit]):")
for k, g, h, corr in out.trace[:8]:
    print(f"  #{k:<3d} g={g:6.3f} h={h:6.3f}  {[(c, dd) for c, dd, _ in corr]}")

print("\nOUTCOME:", out.message)
if out.status == "solved":
    print("\nCorrection log (cell, read -> new, cost):")
    for c in out.corrections:
        print(f"  r{c['row'] + 1}c{c['col'] + 1}: {c['read']} -> {c['new']}   cost {c['cost']:.3f}")
    show(out.solution, "SOLUTION:")
    print("exact match with the true solution:", out.solution == sol)

# ----------------------------------------------------------------- CSP with / without FC / AC-3
print("\nCSP pruning on the corrected puzzle (first reading with the corrections applied):")
grid_fixed = list(rd.grid.reshape(-1))
for c in out.corrections:
    grid_fixed[c["cell"]] = c["new"]
if out.status != "solved":
    grid_fixed = puz  # show pruning on the true puzzle instead
csp = {}
for name, kw in (("BT", dict(fc=False)), ("BT+FC", dict(fc=True)), ("BT+FC+AC3", dict(fc=True, ac3_on=True))):
    r = solve(grid_fixed, max_assignments=500000, **kw)
    csp[name] = r.stats
    print(f"  {name:10s} assignments={r.stats.assignments:7d} backtracks={r.stats.backtracks:7d} "
          f"ac3 revisions={r.stats.revisions:6d}")

# ----------------------------------------------------------------- NN on vs off
print("\nSYSTEM with NN correction ON vs OFF:")
off = correct_and_solve(rd.grid, rd.probs, mode="none", goal=a.goal)
on = res["astar"]
for name, o in (("OFF", off), ("ON ", on)):
    ok = "exact" if o.solution == sol else ("wrong" if o.solution else "-")
    print(f"  NN-correction {name}: status={o.status:10s} solution={ok:5s} csp_calls={o.csp_calls}")
if on.status == "unreadable":
    print("  ->", on.message)

# ----------------------------------------------------------------- plots
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(inst.mosaic, cmap="gray"); ax[0].set_title("input: handwritten mosaic"); ax[0].axis("off")
for i in wrong:
    ax[0].add_patch(Rectangle(((i % 9) * 28, (i // 9) * 28), 28, 28, fill=False, ec="red", lw=2))
draw_grid(ax[1], rd.grid, "NN first reading (red = misread)", red=set(wrong))
if out.status == "solved":
    draw_grid(ax[2], out.solution, "solution (green = corrected cell)", green={c["cell"] for c in out.corrections},
              given=[bool(v) for v in rd.grid.reshape(-1)])
else:
    ax[2].axis("off"); ax[2].text(.5, .5, "UNREADABLE\nleast confident cells:\n" + "\n".join(
        f"r{c['row'] + 1}c{c['col'] + 1} read {c['read']} ({c['confidence']:.2f})" for c in out.least_confident),
        ha="center", va="center", fontsize=11, color="darkred")
fig.tight_layout(); fig.savefig("results/demo_overview.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for algo, o in res.items():
    ax[0].plot([t[0] for t in o.trace], [t[1] for t in o.trace], marker=".", label=algo)
ax[0].set_xlabel("goal-check order"); ax[0].set_ylabel("g (cost so far) of popped node"); ax[0].set_title("search trace"); ax[0].legend()
w = .27; xs = np.arange(3)
ax[1].bar(xs - w / 2, [res[k].nodes_expanded for k in res], w, label="expanded")
ax[1].bar(xs + w / 2, [res[k].nodes_generated for k in res], w, label="generated")
ax[1].set_xticks(xs); ax[1].set_xticklabels(list(res)); ax[1].set_yscale("log"); ax[1].set_title("search effort"); ax[1].legend()
ax[2].bar(xs - w / 2, [max(csp[k].assignments, .5) for k in csp], w, label="assignments")
ax[2].bar(xs + w / 2, [max(csp[k].backtracks, .5) for k in csp], w, label="backtracks")
ax[2].set_xticks(xs); ax[2].set_xticklabels(list(csp)); ax[2].set_yscale("log"); ax[2].set_title("CSP pruning: BT / FC / AC-3"); ax[2].legend()
fig.tight_layout(); fig.savefig("results/demo_search_csp.png", dpi=120); plt.close(fig)
print("\nplots: results/demo_overview.png, results/demo_search_csp.png")
