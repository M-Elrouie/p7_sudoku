"""The three experiments (+ aggregation, tables, plots). Seeds control puzzle + mosaic generation."""
from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .csp import solve
from .generator import ImagePools, build_instance, make_puzzle
from .integration import correct_and_solve, read_mosaic


# ---------------------------------------------------------------- aggregation
def summarize(rows: List[dict], group: List[str], metrics: List[str], seed_key="seed") -> List[dict]:
    """Mean and std *across seeds* of the per-seed mean of each metric."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[tuple(r[k] for k in group)][r[seed_key]].append(r)
    out = []
    for key, seeds in sorted(by.items()):
        rec = dict(zip(group, key))
        for m in metrics:
            per_seed = [np.mean([float(x[m]) for x in rs]) for rs in seeds.values()]
            rec[m + "_mean"], rec[m + "_std"] = float(np.mean(per_seed)), float(np.std(per_seed, ddof=1) if len(per_seed) > 1 else 0.0)
        rec["n_seeds"] = len(seeds)
        out.append(rec)
    return out


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def write_md(path, rows, group, metrics, title):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"### {title}\n\n| " + " | ".join(group + metrics) + " |\n|" + "---|" * (len(group) + len(metrics)) + "\n")
        for r in rows:
            f.write("| " + " | ".join([str(r[g]) for g in group] +
                                      [f"{r[m + '_mean']:.3f} ± {r[m + '_std']:.3f}" for m in metrics]) + " |\n")


def plot_lines(rows, xkey, series_key, metric, ylabel, title, path, logy=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in sorted({r[series_key] for r in rows}):
        rs = sorted([r for r in rows if r[series_key] == s], key=lambda r: r[xkey])
        ax.errorbar([r[xkey] for r in rs], [r[metric + "_mean"] for r in rs],
                    yerr=[r[metric + "_std"] for r in rs], marker="o", capsize=3, label=str(s))
    ax.set_xlabel(xkey); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    ax.set_xticks(sorted({r[xkey] for r in rows}))
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------- experiment 1 & 3
def _instances(model, pools, seed, n, givens, levels):
    """Same base puzzle for every misread level (only the injected images differ)."""
    rng = np.random.default_rng(10_000 + seed)
    for idx in range(n):
        puz, sol = make_puzzle(rng, givens)
        for lv in levels:
            r2 = np.random.default_rng([seed, idx, lv])
            inst = build_instance(r2, pools, puz, sol, lv)
            yield idx, lv, inst, read_mosaic(model, inst.mosaic)


def run_exp1(model, pools, seeds, n, givens=30, levels=range(5), goal="solvable", log=print):
    rows = []
    for s in seeds:
        t0 = time.time()
        for idx, lv, inst, rd in _instances(model, pools, s, n, givens, levels):
            actual = int((rd.grid.reshape(-1) != np.array(inst.puzzle)).sum())
            for algo in ("ucs", "greedy", "astar"):
                o = correct_and_solve(rd.grid, rd.probs, mode="nn", algo=algo, goal=goal)
                rows.append(dict(seed=s, idx=idx, level=lv, algo=algo, actual_misreads=actual,
                                 expanded=o.nodes_expanded, generated=o.nodes_generated,
                                 recovered=int(o.solution == inst.solution),
                                 solved=int(o.status == "solved"),
                                 cost=sum(c["cost"] for c in o.corrections)))
        log(f"  exp1 seed {s} done ({time.time() - t0:.0f}s)")
    return rows


def run_exp3(model, pools, seeds, n, givens=30, levels=range(5), goal="solvable", log=print):
    rows = []
    for s in seeds:
        t0 = time.time()
        for idx, lv, inst, rd in _instances(model, pools, s, n, givens, levels):
            for mode in ("nn", "none", "random"):
                o = correct_and_solve(rd.grid, rd.probs, mode=mode, algo="astar", goal=goal,
                                      rng=np.random.default_rng([s, idx, lv, 7]))
                rows.append(dict(seed=s, idx=idx, level=lv, mode=mode,
                                 recovered=int(o.solution == inst.solution), solved=int(o.status == "solved"),
                                 csp_calls=o.csp_calls, n_corrections=len(o.corrections)))
        log(f"  exp3 seed {s} done ({time.time() - t0:.0f}s)")
    return rows


# ---------------------------------------------------------------- experiment 2
VARIANTS = {"BT": dict(fc=False, ac3_on=False), "BT+FC": dict(fc=True, ac3_on=False),
            "BT+FC+AC3": dict(fc=True, ac3_on=True)}


def run_exp2(seeds, n, givens_levels=(25, 30, 35), cap=100_000, log=print):
    rows = []
    for s in seeds:
        rng = np.random.default_rng(20_000 + s)
        t0 = time.time()
        for gv in givens_levels:
            for idx in range(n):
                puz, _ = make_puzzle(rng, gv)
                for name, kw in VARIANTS.items():
                    t = time.perf_counter()
                    r = solve(puz, max_assignments=cap, **kw)
                    rows.append(dict(seed=s, idx=idx, givens=gv, variant=name,
                                     assignments=r.stats.assignments, backtracks=r.stats.backtracks,
                                     revisions=r.stats.revisions, aborted=int(r.stats.aborted),
                                     ms=1000 * (time.perf_counter() - t)))
        log(f"  exp2 seed {s} done ({time.time() - t0:.0f}s)")
    return rows


# ---------------------------------------------------------------- driver
def run_all(which, model, pools, outdir, seeds, n13, n2, goal="solvable", log=print):
    os.makedirs(outdir, exist_ok=True)
    if 1 in which:
        rows = run_exp1(model, pools, seeds, n13, goal=goal, log=log)
        write_csv(f"{outdir}/exp1_raw.csv", rows)
        g = ["algo", "level"]; m = ["expanded", "generated", "recovered", "actual_misreads"]
        sm = summarize(rows, g, m); write_csv(f"{outdir}/exp1_summary.csv", sm)
        write_md(f"{outdir}/exp1_summary.md", sm, g, m, "Exp 1: UCS vs Greedy vs A* (30 givens)")
        plot_lines(sm, "level", "algo", "expanded", "nodes expanded", "Exp 1: search effort", f"{outdir}/exp1_nodes.png", logy=True)
        plot_lines(sm, "level", "algo", "recovered", "exact-solution recovery rate", "Exp 1: recovery", f"{outdir}/exp1_recovery.png")
    if 2 in which:
        rows = run_exp2(seeds, n2, log=log)
        write_csv(f"{outdir}/exp2_raw.csv", rows)
        g = ["variant", "givens"]; m = ["assignments", "backtracks", "revisions", "aborted", "ms"]
        sm = summarize(rows, g, m); write_csv(f"{outdir}/exp2_summary.csv", sm)
        write_md(f"{outdir}/exp2_summary.md", sm, g, m, "Exp 2: BT vs BT+FC vs BT+FC+AC-3")
        plot_lines(sm, "givens", "variant", "assignments", "assignments (log)", "Exp 2: CSP effort", f"{outdir}/exp2_assignments.png", logy=True)
        plot_lines(sm, "givens", "variant", "backtracks", "backtracks (log)", "Exp 2: backtracks", f"{outdir}/exp2_backtracks.png", logy=True)
    if 3 in which:
        rows = run_exp3(model, pools, seeds, n13, goal=goal, log=log)
        write_csv(f"{outdir}/exp3_raw.csv", rows)
        g = ["mode", "level"]; m = ["recovered", "csp_calls"]
        sm = summarize(rows, g, m); write_csv(f"{outdir}/exp3_summary.csv", sm)
        write_md(f"{outdir}/exp3_summary.md", sm, g, m, "Exp 3: NN-guided vs none vs random correction")
        plot_lines(sm, "level", "mode", "recovered", "exact-solution recovery rate", "Exp 3: recovery", f"{outdir}/exp3_recovery.png")
        plot_lines(sm, "level", "mode", "csp_calls", "CSP calls", "Exp 3: CSP calls", f"{outdir}/exp3_csp_calls.png", logy=True)
