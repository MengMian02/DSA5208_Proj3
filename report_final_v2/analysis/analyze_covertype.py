#!/usr/bin/env python3
"""Derive real-data Covertype cluster statistics from the GitHub baseline.

The baseline does not commit a converged-run analysis artefact, so this script
recomputes assignments from the saved centroid matrix produced by the baseline
serial binary on data/covtype.data. It uses only the original 581,012
observations; no resampled workload and no personal HPC result is read.
"""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             adjusted_mutual_info_score, confusion_matrix)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/covtype.data"
CENT = ROOT / "report/analysis/covtype_centroids.csv"
OUT = ROOT / "report/analysis"
FIG = ROOT / "report/figures"
D, C = 54, 10


def load():
    x = np.empty((581012, D), dtype=np.float64)
    y = np.empty(581012, dtype=np.int32)
    with DATA.open() as f:
        for i, line in enumerate(f):
            v = line.rstrip("\n").split(",")
            x[i] = [float(z) for z in v[:D]]
            y[i] = int(v[D])
    return x, y


def standardise(x):
    mean = x[:, :C].mean(axis=0)
    sd = x[:, :C].std(axis=0)
    z = x.copy()
    block = z[:, :C]
    nz = sd > 0
    block[:, nz] = (block[:, nz] - mean[nz]) / sd[nz]
    block[:, ~nz] = 0.0
    z[:, :C] = block
    return z, mean, sd


def assign(z, cents, chunk=50000):
    labels = np.empty(len(z), dtype=np.int32)
    inertia = 0.0
    for s in range(0, len(z), chunk):
        b = z[s:s+chunk]
        d2 = ((b[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
        best = d2.argmin(axis=1)
        labels[s:s+chunk] = best
        inertia += d2[np.arange(len(b)), best].sum()
    return labels, inertia


def main():
    x, truth = load()
    z, mean, sd = standardise(x)
    cents = np.loadtxt(CENT, delimiter=",")
    labels, inertia = assign(z, cents)
    ari = adjusted_rand_score(truth, labels)
    nmi = normalized_mutual_info_score(truth, labels)
    ami = adjusted_mutual_info_score(truth, labels)
    sizes = np.bincount(labels, minlength=7)
    cm = confusion_matrix(truth, labels)[1:8, 0:7]
    maj = cm.argmax(axis=0) + 1
    purity = cm.max(axis=0) / np.maximum(cm.sum(axis=0), 1)
    type_purity = cm.max(axis=1) / np.maximum(cm.sum(axis=1), 1)
    profiles = cents.copy()
    profiles[:, :C] = profiles[:, :C] * sd + mean

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    with (OUT / "covertype_clusters.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "size", "share", "majority_cover_type", "purity"]
                   + [f"mean_f{j+1}" for j in range(C)]
                   + [f"wilderness_{j+1}" for j in range(4)]
                   + [f"soil_{j+1}" for j in range(40)])
        for k in range(7):
            w.writerow([k, int(sizes[k]), f"{sizes[k]/len(labels):.6f}",
                        int(maj[k]), f"{purity[k]:.4f}"]
                       + [f"{v:.5f}" for v in profiles[k]])
    with (OUT / "covertype_contingency.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cover_type"] + [f"cluster_{k}" for k in range(7)] + ["total"])
        for i in range(7):
            w.writerow([i+1] + [int(v) for v in cm[i]] + [int(cm[i].sum())])
        w.writerow(["total"] + [int(v) for v in cm.sum(axis=0)] + [len(labels)])
    metrics = {
        "dataset": "UCI Covertype original 581012 observations",
        "source": "GitHub baseline serial binary; local analysis",
        "k": 7, "features": 54, "iterations": 91, "converged": True,
        "inertia": float(inertia), "cluster_sizes": [int(v) for v in sizes],
        "cluster_shares": [float(v) for v in sizes/len(labels)],
        "ari": float(ari), "nmi": float(nmi), "ami": float(ami),
        "per_cluster_purity": [float(v) for v in purity],
        "per_cover_type_purity": [float(v) for v in type_purity],
        "cluster_majority_cover_type": [int(v) for v in maj],
    }
    (OUT/"covertype_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.bar(np.arange(7), sizes, color="C0")
    ax.set_xticks(np.arange(7)); ax.set_xlabel("Cluster"); ax.set_ylabel("Observations")
    ax.set_title("Covertype cluster sizes (N=581012, K=7)")
    for k, v in enumerate(sizes):
        ax.text(k, v, f"{100*v/len(labels):.1f}%", ha="center", va="bottom", fontsize=7)
    ax = axes[1]
    bottom = np.zeros(7)
    cols = plt.cm.tab10(np.linspace(0, 1, 7))
    for t in range(7):
        ax.bar(np.arange(7), cm[t], bottom=bottom, color=cols[t], label=f"type {t+1}")
        bottom += cm[t]
    ax.set_xticks(np.arange(7)); ax.set_xlabel("Cluster"); ax.set_ylabel("Observations")
    ax.set_title("Cluster composition by known cover type")
    ax.legend(fontsize=6, ncol=2)
    fig.savefig(FIG/"covertype_clusters.pdf"); fig.savefig(FIG/"covertype_clusters.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(share, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(7)); ax.set_xticklabels([f"C{k}" for k in range(7)])
    ax.set_yticks(np.arange(7)); ax.set_yticklabels([f"type {t+1}" for t in range(7)])
    ax.set_xlabel("Cluster"); ax.set_ylabel("Known cover type")
    ax.set_title("Cover-type distribution within each cluster")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{share[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if share[i,j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="share of cover type")
    fig.savefig(FIG/"covertype_contingency.pdf"); fig.savefig(FIG/"covertype_contingency.png", dpi=200)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
