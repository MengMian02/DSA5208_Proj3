#!/usr/bin/env python3
"""E9 - real-data analysis of the converged K-means run on original Covertype.

Only data/covtype.data (581,012 real observations) is used. Resampled benchmark
workloads are never read here. The centroid matrix comes from the converged
serial run (46 iterations, inertia 3722934.178255).
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             adjusted_mutual_info_score, confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/covtype.data"
CENTROIDS = ROOT / "results/vanda_harry/covtype_converged_1552669.stdct-mgmt-02/centroids.csv"
OUT = ROOT / "results/vanda_harry"
FIGS = ROOT / "figures/vanda_harry"
D = 54
CONTINUOUS = 10


def load_data():
    x = np.empty((581012, D), dtype=np.float64)
    y = np.empty(581012, dtype=np.int32)
    with DATA.open() as handle:
        for i, line in enumerate(handle):
            values = line.rstrip("\n").split(",")
            x[i] = [float(v) for v in values[:D]]
            y[i] = int(values[D])
    return x, y


def standardise(x):
    mean = x[:, :CONTINUOUS].mean(axis=0)
    sd = x[:, :CONTINUOUS].std(axis=0)  # population sd, matches scale_continuous
    z = x.copy()
    block = z[:, :CONTINUOUS]
    nz = sd > 0
    block[:, nz] = (block[:, nz] - mean[nz]) / sd[nz]
    block[:, ~nz] = 0.0
    z[:, :CONTINUOUS] = block
    return z, mean, sd


def assign(z, centroids, chunk=50000):
    labels = np.empty(len(z), dtype=np.int32)
    inertia = 0.0
    for start in range(0, len(z), chunk):
        block = z[start:start + chunk]
        d2 = ((block[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        best = d2.argmin(axis=1)
        labels[start:start + chunk] = best
        inertia += d2[np.arange(len(block)), best].sum()
    return labels, inertia


def main():
    x, truth = load_data()
    z, mean, sd = standardise(x)
    centroids = np.loadtxt(CENTROIDS, delimiter=",")
    assert centroids.shape == (7, D), centroids.shape
    labels, inertia = assign(z, centroids)

    ari = adjusted_rand_score(truth, labels)
    nmi = normalized_mutual_info_score(truth, labels)
    ami = adjusted_mutual_info_score(truth, labels)
    sizes = np.bincount(labels, minlength=7)
    raw = confusion_matrix(truth, labels)  # truth 0..7, cluster 0..6
    contingency = raw[1:8, 0:7]           # truth 1..7 only
    cluster_majority = contingency.argmax(axis=0) + 1
    cluster_purity = (contingency.max(axis=0)
                      / np.maximum(contingency.sum(axis=0), 1))
    type_purity = contingency.max(axis=1) / np.maximum(contingency.sum(axis=1), 1)
    majority_cluster = contingency.argmax(axis=1)

    profiles = centroids.copy()
    profiles[:, :CONTINUOUS] = profiles[:, :CONTINUOUS] * sd + mean

    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    with (OUT / "covertype_clusters.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = (["cluster", "size", "share", "majority_cover_type", "purity"]
                  + ["mean_f%d" % (j + 1) for j in range(CONTINUOUS)]
                  + ["wilderness_%d" % (j + 1) for j in range(4)]
                  + ["soil_%d" % (j + 1) for j in range(40)])
        writer.writerow(header)
        for k in range(7):
            writer.writerow([k, int(sizes[k]), "%.6f" % (sizes[k] / len(labels)),
                             int(cluster_majority[k]), "%.4f" % cluster_purity[k]]
                            + ["%.5f" % v for v in profiles[k]])

    with (OUT / "covertype_contingency.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cover_type"] + ["cluster_%d" % k for k in range(7)] + ["total"])
        for i in range(7):
            writer.writerow([i + 1] + [int(v) for v in contingency[i]]
                            + [int(contingency[i].sum())])
        writer.writerow(["total"] + [int(v) for v in contingency.sum(axis=0)]
                        + [len(labels)])

    metrics = {
        "dataset": "UCI Covertype (original, 581012 real observations)",
        "k": 7, "features": 54, "iterations": 46, "converged": True,
        "inertia": float(inertia),
        "cluster_sizes": [int(v) for v in sizes],
        "cluster_shares": [float(v) for v in sizes / len(labels)],
        "ari": float(ari), "nmi": float(nmi), "ami": float(ami),
        "per_cover_type_purity": [float(v) for v in type_purity],
        "per_cluster_purity": [float(v) for v in cluster_purity],
        "cluster_majority_cover_type": [int(v) for v in cluster_majority],
        "majority_cluster_per_cover_type": [int(v) for v in majority_cluster],
        "note": "Euclidean K-means on 10 z-scored continuous + 44 binary features; "
                "no resampled data used.",
    }
    (OUT / "covertype_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.bar(np.arange(7), sizes, color="C0")
    ax.set_xticks(np.arange(7))
    ax.set_xlabel("K-means cluster")
    ax.set_ylabel("observations")
    ax.set_title("Covertype: cluster size distribution (N=581012, K=7)")
    for k, v in enumerate(sizes):
        ax.text(k, v, "%.1f%%" % (100 * v / len(labels)), ha="center", va="bottom", fontsize=7)
    ax = axes[1]
    bottom = np.zeros(7)
    colours = plt.cm.tab10(np.linspace(0, 1, 7))
    for t in range(7):
        ax.bar(np.arange(7), contingency[t], bottom=bottom, color=colours[t],
               label="cover type %d" % (t + 1))
        bottom += contingency[t]
    ax.set_xticks(np.arange(7))
    ax.set_xlabel("K-means cluster")
    ax.set_ylabel("observations")
    ax.set_title("Cluster composition by known cover type")
    ax.legend(fontsize=6, ncol=2)
    fig.savefig(FIGS / "covertype_clusters.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    share = contingency / np.maximum(contingency.sum(axis=1, keepdims=True), 1)
    image = ax.imshow(share, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(7)); ax.set_xticklabels(["C%d" % k for k in range(7)])
    ax.set_yticks(np.arange(7)); ax.set_yticklabels(["type %d" % (t + 1) for t in range(7)])
    ax.set_xlabel("K-means cluster")
    ax.set_ylabel("known cover type")
    ax.set_title("Covertype: cover-type distribution within each cluster")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, "%.2f" % share[i, j], ha="center", va="center",
                    fontsize=7, color="white" if share[i, j] > 0.5 else "black")
    fig.colorbar(image, ax=ax, label="share of cover type")
    fig.savefig(FIGS / "covertype_contingency.png")
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print("wrote", OUT / "covertype_clusters.csv")
    print("wrote", OUT / "covertype_contingency.csv")
    print("wrote", FIGS / "covertype_clusters.png")
    print("wrote", FIGS / "covertype_contingency.png")


if __name__ == "__main__":
    main()
