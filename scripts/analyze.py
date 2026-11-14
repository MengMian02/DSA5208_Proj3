#!/usr/bin/env python3
"""Analysis and figures for E1-E6.

Reads results/*.csv (written by scripts/run_e1.py, run_mpi.py, run_metal.py),
aggregates the repetitions by median, and writes publication-quality figures to
figures/. An experiment with no measured rows produces a placeholder figure
that says PENDING, so an unexecuted experiment can never be mistaken for a
measured one.

Usage:  python3 scripts/analyze.py [--figures DIR] [--results DIR]
"""


import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplcache"))
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as import_error:
    # The Atlas login node has neither pandas nor matplotlib (system python3 is
    # 3.6.8 and the python3.7 module ships no site-packages), so make the
    # failure self-explanatory instead of dying on an import.
    sys.stderr.write(
        "ERROR: %s\n\n"
        "scripts/analyze.py needs pandas, numpy and matplotlib. Run it on a\n"
        "machine that has them and keep the cluster work on the cluster:\n\n"
        "    scp -r E0550588@atlas8.nus.edu.sg:dsa5208/results/ results/atlas/\n"
        "    python3 scripts/analyze.py --results results/atlas \\\n"
        "        --figures figures/atlas --experiments e1,e2,e3,e4\n" % import_error
    )
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

STYLE = {
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
}


def load(name):
    path = RESULTS / name
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    return frame if len(frame) else None


def summarise(frame, key, value):
    grouped = frame.groupby(key)[value]
    return pd.DataFrame({
        "median": grouped.median(),
        "low": grouped.min(),
        "high": grouped.max(),
        "n": grouped.count(),
    }).reset_index()


def pending(ax, title, detail="no measured data in results/"):
    ax.text(0.5, 0.5, "PENDING\n" + detail, ha="center", va="center",
            fontsize=12, color="crimson", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def display_path(path):
    """Short path for logging; tolerates a relative --results/--figures argument."""
    try:
        return path.resolve().relative_to(ROOT)
    except ValueError:
        return path


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print("wrote", display_path(path))


def error_bars(summary):
    return [summary["median"] - summary["low"], summary["high"] - summary["median"]]


# ---------------------------------------------------------------- E1
def figure_e1(frames):
    serial = frames.get("e1_serial")
    openmp = frames.get("e1_openmp")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    plotted = False
    for frame, label, colour in ((serial, "serial, original", "C0"),
                                 (openmp, "OpenMP, original", "C1")):
        if frame is None:
            continue
        part = frame[(frame["experiment"] == "E1-E1A") | (frame["experiment"] == "E1A")]
        if part.empty:
            continue
        summary = summarise(part, "rows", "fit_s")
        ax.errorbar(summary["rows"], summary["median"], yerr=error_bars(summary),
                    marker="o", capsize=3, color=colour, label=label)
        slope = np.polyfit(np.log(summary["rows"]), np.log(summary["median"]), 1)[0]
        ax.annotate("%s fitted exponent %.2f" % (label, slope), xy=(0.97, 0.18),
                    xycoords="axes fraction", ha="right", color=colour)
        plotted = True
    if serial is not None:
        part = serial[(serial["experiment"] == "E1-E1C") | (serial["experiment"] == "E1C")]
        if not part.empty:
            summary = summarise(part, "rows", "fit_s")
            ax.errorbar(summary["rows"], summary["median"], yerr=error_bars(summary),
                        marker="D", capsize=3, color="C2", linestyle="--",
                        label="serial, fixed-total resampled")
            slope = np.polyfit(np.log(summary["rows"]), np.log(summary["median"]), 1)[0]
            ax.annotate("resampled exponent %.2f" % slope, xy=(0.97, 0.08),
                        xycoords="axes fraction", ha="right", color="C2")
            plotted = True
    if plotted:
        rows = np.array([50_000, 581_012])
        # Reference slope of a perfectly linear (O(N)) fit, anchored at the
        # smallest measured point of the serial curve.
        base = summarise((serial if serial is not None else openmp), "rows", "fit_s")
        anchor = base.iloc[0]
        ax.plot(rows, anchor["median"] * rows / anchor["rows"], "k--", lw=1,
                label="O(N) reference")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("observations N")
        ax.set_ylabel("fit time (s, median of repetitions)")
        ax.set_title("E1A: serial fit time vs N (K=7, D=54, I=30)")
        ax.legend()
    else:
        pending(ax, "E1A: fit time vs N")
    ax = axes[1]
    if serial is not None:
        part = serial[(serial["experiment"] == "E1-E1B") | (serial["experiment"] == "E1B")]
        if not part.empty:
            summary = summarise(part, "k", "fit_s")
            ax.errorbar(summary["k"], summary["median"], yerr=error_bars(summary),
                        marker="s", capsize=3, color="C3")
            slope = np.polyfit(np.log(summary["k"]), np.log(summary["median"]), 1)[0]
            ax.annotate("fitted exponent %.2f" % slope, xy=(0.97, 0.18),
                        xycoords="axes fraction", ha="right", color="C3")
            ax.plot(summary["k"], summary["median"].iloc[0] * summary["k"] / summary["k"].iloc[0],
                    "k--", lw=1, label="O(K) reference")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("clusters K")
            ax.set_ylabel("fit time (s, median)")
            ax.set_title("E1B: serial fit time vs K (N=581012)")
            ax.legend()
        else:
            pending(ax, "E1B: fit time vs K")
    else:
        pending(ax, "E1B: fit time vs K")
    save(fig, "e1_serial_scaling.png")

    # Runtime per observation, the normalised view.
    if serial is not None:
        part = serial[(serial["experiment"] == "E1-E1A") | (serial["experiment"] == "E1A")]
        if not part.empty:
            summary = summarise(part, "rows", "fit_s")
            per_obs = summary["median"] / summary["rows"] * 1e6
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.plot(summary["rows"], per_obs, marker="o", color="C0")
            ax.set_xscale("log")
            ax.set_xlabel("observations N")
            ax.set_ylabel("fit time per observation (us)")
            ax.set_title("E1A: normalised cost per observation")
            save(fig, "e1_time_per_observation.png")


# ---------------------------------------------------------------- E2/E5
def strong_baseline(frame):
    base = frame[frame["ranks"] == frame["ranks"].min()]
    return float(base["fit_s"].median())


def figure_e2(frames):
    frame = frames.get("e2_strong")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if frame is None or frame.empty:
        pending(axes[0], "E2: strong-scaling speedup")
        pending(axes[1], "E2: strong-scaling efficiency")
        save(fig, "e2_strong_scaling.png")
        return
    summary = summarise(frame, "ranks", "fit_s").sort_values("ranks")
    t1 = float(summary.loc[summary["ranks"] == summary["ranks"].min(), "median"].iloc[0])
    speedup = t1 / summary["median"]
    efficiency = speedup / summary["ranks"]

    ax = axes[0]
    ax.plot(summary["ranks"], speedup, marker="o", label="measured fit-time speedup")
    ax.plot(summary["ranks"], summary["ranks"], "k--", lw=1, label="ideal")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("speedup vs 1-rank MPI")
    ax.set_title("E2: MPI strong scaling (N=581012, K=7, I=30)")
    ax.legend()

    ax = axes[1]
    ax.plot(summary["ranks"], efficiency * 100, marker="s", color="C3")
    ax.axhline(100, ls="--", lw=1, color="k")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("parallel efficiency (%)")
    ax.set_title("E2: strong-scaling efficiency")
    save(fig, "e2_strong_scaling.png")


def figure_e2_large(frames):
    """Strong scaling on the fixed 10M resampled workload, with the original
    Covertype curve overlaid when both are present."""
    large = frames.get("e2_strong_large")
    original = frames.get("e2_strong")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if large is None or large.empty:
        pending(axes[0], "E2-large: strong scaling (10M)")
        pending(axes[1], "E2-large: efficiency (10M)")
        save(fig, "e2_large_scaling.png")
        return
    series = [("original Covertype (N=581012)", original, "C0", "o"),
              ("fixed-total resampled (N=10M)", large, "C3", "s")]
    ax = axes[0]
    for label, frame, colour, marker in series:
        if frame is None or frame.empty:
            continue
        summary = summarise(frame, "ranks", "fit_s").sort_values("ranks")
        ax.errorbar(summary["ranks"], summary["median"], yerr=error_bars(summary),
                    marker=marker, capsize=3, color=colour, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("fit time (s, median)")
    ax.set_title("Strong scaling: original vs 10M resampled")
    ax.legend(fontsize=7)

    ax = axes[1]
    for label, frame, colour, marker in series:
        if frame is None or frame.empty:
            continue
        summary = summarise(frame, "ranks", "fit_s").sort_values("ranks")
        t1 = float(summary["median"].iloc[0])
        speedup = t1 / summary["median"]
        ax.plot(summary["ranks"], speedup, marker=marker, color=colour, label=label)
    ranks = np.array(sorted(set(np.concatenate([
        frame["ranks"].unique() for frame in (original, large)
        if frame is not None and not frame.empty]))))
    ax.plot(ranks, ranks, "k--", lw=1, label="ideal")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("speedup vs 1-rank MPI")
    ax.set_title("Strong-scaling speedup")
    ax.legend(fontsize=7)
    save(fig, "e2_large_scaling.png")


def figure_e5(frames):
    frame = frames.get("e2_strong")
    hybrid = frames.get("e4_hybrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if frame is None or frame.empty:
        pending(axes[0], "E5: computation vs communication")
        pending(axes[1], "E5: collective fraction")
        save(fig, "e5_breakdown.png")
        return
    phases = ["local_compute_s", "mpi_reduction_s", "centroid_update_s", "thread_reduction_s"]
    labels = ["local compute", "MPI collectives", "centroid update", "OpenMP reduction"]
    summary = frame.groupby("ranks")[phases + ["fit_s"]].median().reset_index()
    summary["residual_s"] = (summary["fit_s"] - summary[phases].sum(axis=1)).clip(lower=0)
    ranks = summary["ranks"].to_numpy()
    width = 0.6
    ax = axes[0]
    bottom = np.zeros(len(summary))
    colours = ["C0", "C3", "C2", "C4", "0.7"]
    stack = summary[phases].to_numpy().T.tolist() + [summary["residual_s"].to_numpy().tolist()]
    for values, label, colour in zip(stack, labels + ["unattributed"], colours):
        ax.bar(ranks, values, bottom=bottom, width=width, label=label, color=colour)
        bottom += np.asarray(values)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("seconds per fit (median)")
    ax.set_title("E5: fitting-time breakdown")
    ax.legend(fontsize=7)

    ax = axes[1]
    fraction = summary["mpi_reduction_s"] / summary["fit_s"]
    ax.plot(ranks, fraction * 100, marker="o", color="C3", label="MPI collective share")
    compute = summary["local_compute_s"] / summary["fit_s"]
    ax.plot(ranks, compute * 100, marker="s", color="C0", label="local compute share")
    imbalance = frame.groupby("ranks")["fit_imbalance_s"].median().to_numpy()
    fit = summary["fit_s"].to_numpy()
    ax.plot(ranks, np.where(fit > 0, imbalance / fit * 100, 0), marker="^", color="C2",
            label="rank imbalance share")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("share of fit time (%)")
    ax.set_title("E5: shares of fitting time (max across ranks)")
    ax.legend(fontsize=7)
    save(fig, "e5_breakdown.png")

    if hybrid is not None and not hybrid.empty:
        # E5 requirement: hybrid fitting time broken down by major phase.
        hybrid = hybrid.copy()
        hybrid["layout"] = hybrid["ranks"].astype(str) + "x" + hybrid["threads"].astype(str)
        phases = ["local_compute_s", "thread_reduction_s", "mpi_reduction_s", "centroid_update_s"]
        labels = ["local compute", "OpenMP fold", "MPI collectives", "centroid update"]
        table = hybrid.groupby("layout")[phases + ["fit_s"]].median()
        table["residual_s"] = (table["fit_s"] - table[phases].sum(axis=1)).clip(lower=0)
        layouts = list(table.index)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        bottom = np.zeros(len(table))
        for column, label, colour in zip(phases + ["residual_s"], labels + ["unattributed"],
                                        ["C0", "C4", "C3", "C2", "0.7"]):
            ax.bar(layouts, table[column].to_numpy(), bottom=bottom, label=label, color=colour)
            bottom += table[column].to_numpy()
        ax.set_ylabel("seconds per fit (median)")
        ax.set_xlabel("ranks x threads")
        ax.set_title("E5: hybrid fitting-time breakdown by phase")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=7)
        save(fig, "e5_hybrid_breakdown.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        for ranks, group in hybrid.groupby("ranks"):
            summary = summarise(group, "threads", "fit_s")
            ax.errorbar(summary["threads"], summary["median"], yerr=error_bars(summary),
                        marker="o", capsize=3, label="%d rank(s)" % ranks)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("OpenMP threads per rank")
        ax.set_ylabel("fit time (s, median)")
        ax.set_title("E4: hybrid breakdown - threads per rank")
        ax.legend(fontsize=7)
        save(fig, "e4_threads_vs_time.png")


# ---------------------------------------------------------------- E3
def figure_e3(frames):
    frame = frames.get("e3_weak")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if frame is None or frame.empty:
        pending(axes[0], "E3: weak-scaling runtime")
        pending(axes[1], "E3: weak-scaling efficiency")
        save(fig, "e3_weak_scaling.png")
        return
    summary = summarise(frame, "ranks", "fit_s").sort_values("ranks")
    reference = float(summary.loc[summary["ranks"] == summary["ranks"].min(), "median"].iloc[0])
    efficiency = reference / summary["median"]

    ax = axes[0]
    ax.errorbar(summary["ranks"], summary["median"], yerr=error_bars(summary),
                marker="o", capsize=3, label="measured")
    ax.axhline(reference, ls="--", lw=1, color="k", label="ideal constant time")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("MPI ranks (50000 generated observations per rank)")
    ax.set_ylabel("fit time (s, median)")
    ax.set_title("E3: MPI weak scaling runtime")
    ax.legend()

    ax = axes[1]
    ax.plot(summary["ranks"], efficiency * 100, marker="s", color="C3")
    ax.axhline(100, ls="--", lw=1, color="k")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("weak-scaling efficiency E(P) = T(1,N0)/T(P,P*N0) (%)")
    ax.set_title("E3: weak-scaling efficiency")
    save(fig, "e3_weak_scaling.png")


# ---------------------------------------------------------------- E4
def figure_e4(frames):
    frame = frames.get("e4_hybrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if frame is None or frame.empty:
        pending(axes[0], "E4: hybrid layouts")
        pending(axes[1], "E4: efficiency at equal cores")
        save(fig, "e4_hybrid.png")
        return
    frame = frame.copy()
    frame["layout"] = frame["ranks"].astype(str) + "x" + frame["threads"].astype(str)
    summary = summarise(frame, "layout", "fit_s")
    summary["ranks"] = [int(v.split("x")[0]) for v in summary["layout"]]
    summary["threads"] = [int(v.split("x")[1]) for v in summary["layout"]]
    summary = summary.sort_values("ranks")

    ax = axes[0]
    colours = plt.cm.viridis(np.linspace(0.1, 0.9, summary["threads"].nunique()))
    lookup = dict(zip(sorted(summary["threads"].unique()), colours))
    for threads, group in summary.groupby("threads"):
        ax.errorbar(group["ranks"], group["median"],
                    yerr=[group["median"] - group["low"], group["high"] - group["median"]],
                    marker="o", capsize=3, color=lookup[threads],
                    label="%d thread(s)/rank" % threads)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("fit time (s, median)")
    ax.set_title("E4: hybrid rank/thread layouts")
    ax.legend(fontsize=7)

    ax = axes[1]
    best = summary["median"].min()
    efficiency = best / summary["median"]
    ax.bar(summary["layout"], efficiency * 100, color="C0")
    ax.axhline(100, ls="--", lw=1, color="k")
    ax.set_xlabel("ranks x threads")
    ax.set_ylabel("efficiency vs best layout (%)")
    ax.set_title("E4: layouts relative to the best measured configuration")
    ax.tick_params(axis="x", rotation=45)
    save(fig, "e4_hybrid.png")


# ---------------------------------------------------------------- E6
def figure_e6(frames):
    frame = frames.get("e6_metal")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if frame is None or frame.empty:
        pending(axes[0], "E6: CPU vs GPU assignment")
        pending(axes[1], "E6: GPU-assisted fitting")
        save(fig, "e6_cpu_vs_gpu.png")
        return
    ax = axes[0]
    # The CPU assignment pass is measured once per run; average it across the
    # stages so a single series is drawn for the CPU.
    cpu = summarise(frame, "rows", "cpu_assign_s").sort_values("rows")
    ax.errorbar(cpu["rows"], cpu["median"], yerr=error_bars(cpu), marker="o",
                capsize=3, color="C0", label="CPU assignment (double)")
    colours = {"A_assign": "C3", "B_full": "C4"}
    for stage, group in frame.groupby("stage"):
        gpu = summarise(group, "rows", "gpu_kernel_s").sort_values("rows")
        ax.errorbar(gpu["rows"], gpu["median"], yerr=error_bars(gpu), marker="s",
                    capsize=3, color=colours.get(stage, "C5"),
                    label="GPU kernel, stage " + str(stage) + " (float)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("observations N")
    ax.set_ylabel("one assignment pass (s, median)")
    ax.set_title("E6: nearest-centroid assignment, CPU vs Metal")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()

    ax = axes[1]
    for stage, group in frame.groupby("stage"):
        summary = summarise(group, "rows", "fit_s").sort_values("rows")
        ax.errorbar(summary["rows"], summary["median"], yerr=error_bars(summary),
                    marker="o", capsize=3, label="GPU-assisted fit %s" % stage)
    floor = float(frame["dispatch_floor_s"].median())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("observations N")
    ax.set_ylabel("fit time (s, median)")
    ax.set_title("E6: GPU-assisted fitting (dispatch floor %.2e s)" % floor)
    ax.legend(fontsize=7)
    save(fig, "e6_cpu_vs_gpu.png")

    agreement = frame["label_agreement"].min()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(["min label agreement"], [agreement * 100], color="C2")
    ax.set_ylim(min(99.0, agreement * 100 - 1), 100.2)
    ax.set_ylabel("agreement with double-precision CPU labels (%)")
    ax.set_title("E6: GPU correctness vs CPU reference")
    save(fig, "e6_validation.png")


def write_summary(frames, requested=None):
    lines = ["# Measured results summary", "",
             "Medians over the recorded repetitions. Generated by scripts/analyze.py.",
             "Every row below is derived from a run that actually executed; nothing is",
             "interpolated or assumed.", ""]
    spec = {
        "e1_serial": ("E1 serial", ["rows", "k"], ["fit_s", "load_s", "preprocess_s", "init_s", "total_s"]),
        "e2_strong": ("E2 MPI strong scaling", ["ranks"], ["fit_s", "local_compute_s", "mpi_reduction_s", "centroid_update_s", "fit_imbalance_s"]),
        "e2_strong_large": ("E2-large MPI strong scaling (10M resampled)", ["ranks"], ["fit_s", "local_compute_s", "mpi_reduction_s", "distribution_s"]),
        "e3_weak": ("E3 MPI weak scaling", ["ranks", "rows"], ["fit_s", "local_compute_s", "mpi_reduction_s", "distribution_s"]),
        "e4_hybrid": ("E4 hybrid", ["ranks", "threads"], ["fit_s", "local_compute_s", "mpi_reduction_s", "thread_reduction_s"]),
        "e6_metal": ("E6 Metal", ["stage", "rows"], ["fit_s", "cpu_assign_s", "gpu_kernel_s", "accumulate_s", "label_agreement"]),
    }
    for key, (title, keys, values) in spec.items():
        if requested and key.split("_")[0] not in requested:
            continue
        frame = frames.get(key)
        lines += ["## " + title, ""]
        if frame is None or frame.empty:
            lines += ["PENDING - no measured data in results/ yet.", ""]
            continue
        table = frame.groupby(keys)[values].median().reset_index()
        header = "| " + " | ".join(keys + values) + " |"
        sep = "|" + "|".join(["---"] * (len(keys) + len(values))) + "|"
        lines += [header, sep]
        for _, row in table.iterrows():
            cells = [str(row[k]) for k in keys]
            cells += ["%.6g" % row[v] for v in values]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    path = RESULTS / "SUMMARY.md"
    path.write_text("\n".join(lines) + "\n")
    print("wrote", display_path(path))


FIGURES_BY_EXPERIMENT = {
    "e1": (("e1_serial", "e1_openmp"), figure_e1),
    "e2": (("e2_strong",), figure_e2),
    "e2large": (("e2_strong_large", "e2_strong"), figure_e2_large),
    "e3": (("e3_weak",), figure_e3),
    "e4": (("e4_hybrid",), figure_e4),
    "e5": (("e2_strong", "e4_hybrid"), figure_e5),
    "e6": (("e6_metal",), figure_e6),
}
ALL_SERIES = ("e1_serial", "e1_openmp", "e2_strong", "e2_strong_large",
              "e3_weak", "e4_hybrid", "e6_metal")


def main():
    global RESULTS, FIGURES
    parser = argparse.ArgumentParser(description="Generate E1-E6 figures")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--experiments", type=str, default="",
                        help="comma separated subset, e.g. e1,e2,e3,e4 (default: all)")
    args = parser.parse_args()
    RESULTS = args.results
    FIGURES = args.figures
    plt.rcParams.update(STYLE)

    requested = [value.strip().lower() for value in args.experiments.split(",") if value.strip()]
    if not requested:
        requested = list(FIGURES_BY_EXPERIMENT)
    unknown = [name for name in requested if name not in FIGURES_BY_EXPERIMENT]
    if unknown:
        parser.error("unknown experiment(s): %s" % ", ".join(unknown))

    frames = {name: load(name + ".csv") for name in ALL_SERIES}
    for name in requested:
        FIGURES_BY_EXPERIMENT[name][1](frames)
    write_summary(frames, requested)
    print("figures written to", FIGURES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
