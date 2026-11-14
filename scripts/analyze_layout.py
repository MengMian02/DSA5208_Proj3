#!/usr/bin/env python3
"""Plot the two committed 256-core Vanda node layouts.

These CSVs are historical measurements; the two jobs landed on different node
families (GN-A40 vs CN), so the figure is labelled hardware-confounded and is
not presented as a controlled node-count experiment. The repaired E7 job is the
controlled rerun on one CN allocation.
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures/vanda_harry/analysis"
CASES = [
    ("4x64 GN-A40", ROOT / "results/vanda/vanda_e2_strong_1551056.stdct-mgmt-02.csv", "C0", "o"),
    ("8x32 CN", ROOT / "results/vanda/vanda_e2_strong_8x32_1551069.stdct-mgmt-02.csv", "C3", "s"),
]


def summarise(path):
    groups = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            groups[int(row["ranks"])].append(float(row["fit_s"]))
    ranks = sorted(groups)
    return np.array(ranks), np.array([statistics.median(groups[r]) for r in ranks])


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, path, colour, marker in CASES:
        ranks, fit = summarise(path)
        base = fit[0]
        ax = axes[0]
        ax.plot(ranks, base / fit, marker=marker, color=colour, label=label)
        ax = axes[1]
        ax.plot(ranks, (base / fit) / ranks * 100, marker=marker, color=colour, label=label)
    axes[0].plot([1, 256], [1, 256], "k--", lw=1, label="ideal")
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log", base=2)
    axes[0].set_xlabel("MPI ranks"); axes[0].set_ylabel("speedup vs 1-rank MPI")
    axes[0].set_title("Committed Vanda layouts (hardware-confounded)")
    axes[0].legend(fontsize=7)
    axes[1].axhline(100, ls="--", lw=1, color="k")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("MPI ranks"); axes[1].set_ylabel("parallel efficiency (%)")
    axes[1].set_title("Efficiency: same caveat")
    axes[1].legend(fontsize=7)
    fig.savefig(FIGS / "e7_existing_layouts.png")
    print("wrote", FIGS / "e7_existing_layouts.png")


if __name__ == "__main__":
    main()
