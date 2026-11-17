#!/usr/bin/env python3
"""Build every report table and figure from committed GitHub-baseline CSVs.

Canonical files are the .stdct-mgmt-02 copies; byte-identical duplicates in the
baseline are not double-counted. No personal HPC result is read.
"""
from __future__ import annotations
import csv, json, math, statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report"
TAB = OUT / "tables"
FIG = OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.autolayout": True, "savefig.dpi": 200})

SER = ROOT/"results/vanda/vanda_e1_serial_1546646.stdct-mgmt-02.csv"
E2A = ROOT/"results/vanda/vanda_e2_strong_1551056.stdct-mgmt-02.csv"
E2B = ROOT/"results/vanda/vanda_e2_strong_8x32_1551069.stdct-mgmt-02.csv"
E3A = ROOT/"results/vanda/vanda_e3_weak_1551057.stdct-mgmt-02.csv"
E3B = ROOT/"results/vanda/vanda_e3_weak_8x32_1551070.stdct-mgmt-02.csv"
H64 = ROOT/"results/vanda/vanda_e4_hybrid72_1551076.stdct-mgmt-02.csv"
H256 = ROOT/"results/vanda/vanda_e4_hybrid256_1554129.stdct-mgmt-02.csv"
H8 = ROOT/"results/vanda/vanda_e4_hybrid_8x32_1554130.stdct-mgmt-02.csv"
PIN = ROOT/"results/vanda/e4_pinned_4x64_1556850.csv"
CUDA = ROOT/"results/vanda/e7_cuda_1554475.csv"
METAL = ROOT/"results/e6_metal.csv"
LOCALH = ROOT/"results/e4_hybrid.csv"


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def med(rs, key):
    return statistics.median(float(r[key]) for r in rs)


def group(rs, keys):
    g = defaultdict(list)
    for r in rs:
        g[tuple(int(r[k]) if k in ("ranks", "threads", "rows") else r[k] for k in keys)].append(r)
    return g


def savefig(fig, name):
    fig.savefig(FIG/name)
    fig.savefig(FIG/(Path(name).stem + ".png"), dpi=130)
    plt.close(fig)
    print("wrote", FIG/name)


# ---------- serial
def serial():
    rs = rows(SER)
    g = group(rs, ["experiment", "rows", "k"])
    n_rows, k_rows = [], []
    for (exp, n, k), v in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        if exp.startswith("E1-E1A") and float(k) == 7:
            n_rows.append((n, med(v, "fit_s"), med(v, "total_s")))
        if exp.startswith("E1-E1B"):
            k_rows.append((int(float(k)), med(v, "fit_s"), med(v, "total_s")))
    with (TAB/"serial_N.tex").open("w") as f:
        f.write("\\begin{tabular}{rrrr}\n\\toprule\n$N$ & $K$ & fit (s) & total (s)\\\\\n\\midrule\n")
        for n, fit, total in n_rows:
            f.write(f"{n:,} & 7 & {fit:.4f} & {total:.4f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    k_rows.sort()
    with (TAB/"serial_K.tex").open("w") as f:
        f.write("\\begin{tabular}{rrr}\n\\toprule\n$K$ & fit (s) & total (s)\\\\\n\\midrule\n")
        for k, fit, total in k_rows:
            f.write(f"{k} & {fit:.4f} & {total:.4f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    x = np.array([r[0] for r in n_rows], float); y = np.array([r[1] for r in n_rows])
    axes[0].plot(x, y, "o-", color="C0")
    axes[0].plot(x, y[0]*(x/x[0]), "k--", lw=1, label="O(N) reference")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Observations $N$"); axes[0].set_ylabel("fit time (s)")
    axes[0].set_title("Serial fit vs $N$ (K=7, I=30)")
    axes[0].legend(fontsize=7)
    slope = np.polyfit(np.log(x), np.log(y), 1)[0]
    axes[0].annotate(f"slope {slope:.2f}", xy=(0.05, 0.85), xycoords="axes fraction", fontsize=8)
    x = np.array([r[0] for r in k_rows], float); y = np.array([r[1] for r in k_rows])
    axes[1].plot(x, y, "s-", color="C3")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Clusters $K$"); axes[1].set_ylabel("fit time (s)")
    axes[1].set_title("Serial fit vs $K$ (N=581012, I=30)")
    slope = np.polyfit(np.log(x), np.log(y), 1)[0]
    axes[1].annotate(f"slope {slope:.2f}", xy=(0.05, 0.85), xycoords="axes fraction", fontsize=8)
    savefig(fig, "serial.pdf")
    return n_rows, k_rows


# ---------- strong / weak
def scaling(path, kind):
    rs = rows(path)
    g = group(rs, ["ranks"])
    out = []
    for key in sorted(g):
        rank = key[0]
        v = g[key]
        out.append({"ranks": rank, "fit": med(v, "fit_s"),
                    "local": med(v, "local_compute_s"),
                    "coll": med(v, "mpi_reduction_s"),
                    "imbalance": med(v, "fit_imbalance_s"),
                    "rows": int(v[0]["rows"]),
                    "obs": int(v[0]["obs_per_rank"]),
                    "fit_min": min(float(x["fit_s"]) for x in v),
                    "fit_max": max(float(x["fit_s"]) for x in v),
                    "hash": len(set(x["label_hash"] for x in v))})
    base = out[0]["fit"]
    for o in out:
        o["speedup"] = base/o["fit"]
        o["eff"] = o["speedup"]/o["ranks"]
        o["coll_share"] = o["coll"]/o["fit"] if o["fit"] else 0.0
    return out


def strong():
    a = scaling(E2A, "strong"); b = scaling(E2B, "strong")
    with (TAB/"strong.tex").open("w") as f:
        f.write("\\begin{tabular}{llrrrrr}\n\\toprule\nLayout & ranks & fit (s) & speedup & efficiency & collective (s) & collective share\\\\\n\\midrule\n")
        for label, data in (("4$\\times$64", a), ("8$\\times$32", b)):
            for o in data:
                f.write(f"{label} & {o['ranks']} & {o['fit']:.4f} & {o['speedup']:.2f} & {100*o['eff']:.1f}\\% & {o['coll']:.4f} & {100*o['coll_share']:.1f}\\%\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for label, data, c, m in (("4$\\times$64 GN-A40", a, "C0", "o"), ("8$\\times$32 CN", b, "C3", "s")):
        axes[0].plot([o["ranks"] for o in data], [o["speedup"] for o in data], marker=m, color=c, label=label)
        axes[1].plot([o["ranks"] for o in data], [100*o["eff"] for o in data], marker=m, color=c, label=label)
    r = [o["ranks"] for o in a]
    axes[0].plot(r, r, "k--", lw=1, label="ideal")
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log", base=2)
    axes[0].set_xlabel("MPI ranks"); axes[0].set_ylabel("speedup vs 1 rank")
    axes[0].set_title("MPI strong scaling (N=581012)")
    axes[0].legend(fontsize=7)
    axes[1].axhline(100, ls="--", color="k", lw=1)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("MPI ranks"); axes[1].set_ylabel("parallel efficiency (%)")
    axes[1].set_title("Strong-scaling efficiency")
    axes[1].legend(fontsize=7)
    savefig(fig, "strong.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for label, data, c, m in (("4$\\times$64 GN-A40", a, "C0", "o"), ("8$\\times$32 CN", b, "C3", "s")):
        axes[0].plot([o["ranks"] for o in data], [o["coll"] for o in data], marker=m, color=c, label=label)
        axes[1].plot([o["ranks"] for o in data], [100*o["coll_share"] for o in data], marker=m, color=c, label=label)
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
    axes[0].set_xlabel("MPI ranks"); axes[0].set_ylabel("MPI collective time (s)")
    axes[0].set_title("Collective time (K=7, D=54)")
    axes[0].legend(fontsize=7)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("MPI ranks"); axes[1].set_ylabel("share of fit time (%)")
    axes[1].set_title("Collective share of fit")
    axes[1].legend(fontsize=7)
    savefig(fig, "communication.pdf")
    return a, b


def weak():
    a = scaling(E3A, "weak"); b = scaling(E3B, "weak")
    base_a = a[0]["fit"]; base_b = b[0]["fit"]
    with (TAB/"weak.tex").open("w") as f:
        f.write("\\begin{tabular}{llrrrrr}\n\\toprule\nLayout & ranks & total $N$ & fit (s) & weak eff. & collective (s) & collective share\\\\\n\\midrule\n")
        for label, data, base in (("4$\\times$64", a, base_a), ("8$\\times$32", b, base_b)):
            for o in data:
                f.write(f"{label} & {o['ranks']} & {o['rows']:,} & {o['fit']:.4f} & {100*base/o['fit']:.1f}\\% & {o['coll']:.4f} & {100*o['coll_share']:.1f}\\%\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for label, data, base, c, m in (("4$\\times$64 GN-A40", a, base_a, "C0", "o"), ("8$\\times$32 CN", b, base_b, "C3", "s")):
        axes[0].plot([o["ranks"] for o in data], [o["fit"] for o in data], marker=m, color=c, label=label)
        axes[1].plot([o["ranks"] for o in data], [100*base/o["fit"] for o in data], marker=m, color=c, label=label)
    axes[0].set_xscale("log", base=2); axes[0].set_xlabel("MPI ranks")
    axes[0].set_ylabel("fit time (s)"); axes[0].set_title("Weak scaling: 50,000 observations/rank")
    axes[0].legend(fontsize=7)
    axes[1].axhline(100, ls="--", color="k", lw=1)
    axes[1].set_xscale("log", base=2); axes[1].set_xlabel("MPI ranks")
    axes[1].set_ylabel("weak efficiency (%)"); axes[1].set_title("Weak-scaling efficiency")
    axes[1].legend(fontsize=7)
    savefig(fig, "weak.pdf")
    return a, b


# ---------- OpenMP local
def openmp():
    rs = rows(LOCALH)
    g = group(rs, ["ranks", "threads"])
    data = []
    base = med(g[(1, 1)], "fit_s")
    for t in (1, 2, 4):
        v = g[(1, t)]
        fit = med(v, "fit_s")
        data.append((t, fit, base/fit, base/fit/t, med(v, "thread_reduction_s")))
    with (TAB/"openmp.tex").open("w") as f:
        f.write("\\begin{tabular}{rrrrr}\n\\toprule\nthreads & fit (s) & speedup & efficiency & thread reduction (s)\\\\\n\\midrule\n")
        for t, fit, sp, eff, tr in data:
            f.write(f"{t} & {fit:.4f} & {sp:.2f} & {100*eff:.1f}\\% & {tr:.2e}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot([d[0] for d in data], [d[1] for d in data], "o-", color="C0")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("OpenMP threads (one MPI rank)"); ax.set_ylabel("fit time (s)")
    ax.set_title("OpenMP scaling, Apple M1 Pro (N=581012)")
    savefig(fig, "openmp.pdf")
    return data


# ---------- hybrid
def hybrid64():
    rs = rows(H64)
    g = group(rs, ["ranks", "threads"])
    data = []
    base = med(g[(64, 1)], "fit_s")
    for key in sorted(g, key=lambda k: (-k[0], k[1])):
        v = g[key]
        fit = med(v, "fit_s")
        data.append((key[0], key[1], fit, base/fit,
                     med(v, "local_compute_s"), med(v, "mpi_reduction_s"),
                     med(v, "thread_reduction_s")))
    with (TAB/"hybrid64.tex").open("w") as f:
        f.write("\\begin{tabular}{rrrrrrr}\n\\toprule\nranks & threads & fit (s) & speedup vs 64$\\times$1 & local (s) & collective (s) & thread reduction (s)\\\\\n\\midrule\n")
        for r, t, fit, sp, loc, coll, tr in data:
            f.write(f"{r} & {t} & {fit:.4f} & {sp:.2f} & {loc:.4f} & {coll:.5f} & {tr:.2e}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    labels = [f"{r}x{t}" for r, t, *_ in data]
    axes[0].bar(labels, [d[2] for d in data], color="C0")
    axes[0].set_ylabel("fit time (s)"); axes[0].set_title("64-core Hybrid layouts (fixed 64 cores)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].plot(labels, [d[4] for d in data], "o-", color="C0", label="local compute")
    axes[1].plot(labels, [d[5] for d in data], "s-", color="C3", label="MPI collective")
    axes[1].plot(labels, [max(d[6], 1e-7) for d in data], "^-", color="C2", label="OpenMP fold")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("seconds per fit (max across ranks)")
    axes[1].set_title("Phase times (not additive)")
    axes[1].tick_params(axis="x", rotation=45); axes[1].legend(fontsize=7)
    savefig(fig, "hybrid64.pdf")
    return data


def hybrid256():
    pinned = rows(PIN); unpin = rows(H256)
    def stats(rs):
        v = [float(r["fit_s"]) for r in rs]
        return {"n": len(v), "median": statistics.median(v), "mean": statistics.mean(v),
                "min": min(v), "max": max(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0,
                "coll": statistics.median(float(r["mpi_reduction_s"]) for r in rs)}
    p = stats(pinned)
    u256 = stats([r for r in unpin if int(r["ranks"]) == 256])
    e2 = scaling(E2A, "strong")[-1]
    with (TAB/"hybrid256.tex").open("w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\nConfiguration & reps & median (s) & min (s) & max (s) & SD (s) & collective (s)\\\\\n\\midrule\n")
        f.write(f"MPI-only 256 ranks & 3 & {e2['fit']:.4f} & {e2['fit_min']:.4f} & {e2['fit_max']:.4f} & -- & {e2['coll']:.4f}\\\\\n")
        f.write(f"Hybrid 256$\\times$1 pinned & {p['n']} & {p['median']:.4f} & {p['min']:.4f} & {p['max']:.4f} & {p['sd']:.4f} & {p['coll']:.4f}\\\\\n")
        f.write(f"Hybrid 256$\\times$1 unpinned & {u256['n']} & {u256['median']:.4f} & {u256['min']:.4f} & {u256['max']:.4f} & {u256['sd']:.4f} & {u256['coll']:.4f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    pv = [float(r["fit_s"]) for r in pinned]
    uv = [float(r["fit_s"]) for r in unpin if int(r["ranks"]) == 256]
    ax.scatter([1]*len(pv), pv, color="C0", marker="o", label="pinned 256x1")
    ax.scatter([2]*len(uv), uv, color="C3", marker="s", label="unpinned 256x1")
    ax.scatter([0], [e2["fit"]], color="k", marker="D", label="MPI-only 256")
    ax.hlines(statistics.median(pv), 0.75, 1.25, color="C0")
    ax.hlines(statistics.median(uv), 1.75, 2.25, color="C3")
    ax.set_xticks([0, 1, 2], ["MPI-only", "pinned 256x1", "unpinned 256x1"])
    ax.set_yscale("log"); ax.set_ylabel("fit time (s)")
    ax.set_title("256-core Hybrid 256x1: individual repetitions")
    ax.legend(fontsize=7)
    savefig(fig, "hybrid256.pdf")
    return p, u256, e2


# ---------- GPU
def gpu():
    rs = rows(CUDA); g = group(rs, ["stage", "rows"])
    data = []
    for key in sorted(g):
        v = g[key]
        data.append((key[0], key[1], med(v, "fit_s"), med(v, "gpu_kernel_s"),
                     med(v, "cpu_assign_s"), med(v, "upload_s"),
                     med(v, "readback_s"), med(v, "accumulate_s"),
                     min(float(r["label_agreement"]) for r in v)))
    with (TAB/"cuda.tex").open("w") as f:
        f.write("\\begin{tabular}{llrrrrrrr}\n\\toprule\nstage & $N$ & fit (s) & kernel (ms) & CPU assign (s) & upload (s) & readback (s) & accumulate (s) & agreement\\\\\n\\midrule\n")
        for stage, n, fit, ker, cpu, up, rb, acc, agr in data:
            f.write(f"{stage.replace(chr(95), chr(92)+chr(95))} & {n:,} & {fit:.4f} & {1000*ker:.3f} & {cpu:.4f} & {up:.4f} & {rb:.4f} & {acc:.4f} & {agr:.6f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for stage, c, m in (("A_assign", "C3", "o"), ("B_full", "C4", "s")):
        d = [x for x in data if x[0] == stage]
        axes[0].plot([x[1] for x in d], [x[3] for x in d], marker=m, color=c, label=f"GPU kernel {stage}")
        axes[1].plot([x[1] for x in d], [x[2] for x in d], marker=m, color=c, label=f"fit {stage}")
    d = [x for x in data if x[0] == "A_assign"]
    axes[0].plot([x[1] for x in d], [x[4] for x in d], "k--", label="CPU assignment (double)")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Observations $N$"); axes[0].set_ylabel("seconds per assignment pass")
    axes[0].set_title("CUDA A40: kernel vs CPU assignment"); axes[0].legend(fontsize=7)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Observations $N$"); axes[1].set_ylabel("fit time (s)")
    axes[1].set_title("CUDA A40: Stage A vs Stage B"); axes[1].legend(fontsize=7)
    savefig(fig, "cuda.pdf")
    return data


# ---------- covertype table
def covertype():
    m = json.loads((OUT/"analysis/covertype_metrics.json").read_text())
    clusters = rows(OUT/"analysis/covertype_clusters.csv")
    with (TAB/"covertype.tex").open("w") as f:
        f.write("\\begin{tabular}{rrrrrrrr}\n\\toprule\ncluster & size & share & majority type & purity & elevation & road dist. & fire dist.\\\\\n\\midrule\n")
        for c in clusters:
            f.write(f"{int(c['cluster'])} & {int(c['size']):,} & {100*float(c['share']):.1f}\\% & {int(c['majority_cover_type'])} & {float(c['purity']):.3f} & "
                    f"{float(c['mean_f1']):.0f} & {float(c['mean_f6']):.0f} & {float(c['mean_f10']):.0f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return m


if __name__ == "__main__":
    n, k = serial(); a, b = strong(); wa, wb = weak(); om = openmp()
    h64 = hybrid64(); p, u, e2 = hybrid256(); g = gpu(); cm = covertype()
    summary = {"serial_N": n, "serial_K": k, "strong_4x64": a, "strong_8x32": b,
               "weak_4x64": wa, "weak_8x32": wb, "openmp": om,
               "hybrid64": h64, "pinned": p, "unpinned_256x1": u,
               "mpi_only_256": e2, "cuda": g, "covertype": cm}
    (OUT/"analysis"/"report_metrics.json").write_text(json.dumps(summary, indent=2))
    print("all report assets generated")
