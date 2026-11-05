#!/usr/bin/env python3
"""E1 - serial scaling with problem size.

Two grids, with D = 54, fixed-iteration mode, seed 42:

  E1A  N in {50000, 100000, 200000, 400000, 581012}, K = 7
  E1B  K in {2, 4, 7, 10, 16},                       N = 581012

The primary metric is fit_s (Lloyd iterations only). Loading, preprocessing,
k-means++ initialization and the final inertia validation are recorded in their
own columns and never folded into fit_s.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bencher

E1A_ROWS = [50_000, 100_000, 200_000, 400_000, 581_012]
E1B_K = [2, 4, 7, 10, 16]
K = 7
SEED = 42
FULL_ROWS = 581_012


def build_cases(grid):
    cases = []
    if grid in ("e1a", "both"):
        for rows in E1A_ROWS:
            cases.append({"grid": "E1A", "rows": rows, "k": K})
    if grid in ("e1b", "both"):
        for k in E1B_K:
            cases.append({"grid": "E1B", "rows": FULL_ROWS, "k": k})
    return cases


def main():
    parser = argparse.ArgumentParser(description="E1 serial scaling benchmark")
    parser.add_argument("--data", type=Path, default=bencher.ROOT / "data/covtype.data")
    parser.add_argument("--implementation", choices=["serial", "openmp"], default="serial")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--grid", choices=["e1a", "e1b", "both"], default="both")
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--results", type=Path, default=bencher.RESULTS / "e1_serial.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    binary = bencher.require_binary(bencher.ROOT / ("kmeans_" + args.implementation))
    meta = bencher.environment()
    run_id = bencher.make_run_id("e1")
    failures = 0

    for case in build_cases(args.grid):
        for repetition in range(1, args.repetitions + 1):
            command = [str(binary), str(args.data), str(case["k"]), str(args.max_iter),
                       str(SEED), "--fixed-iterations", "--rows", str(case["rows"])]
            env = {"OMP_NUM_THREADS": str(args.threads)}
            label = "%s rows=%d k=%d rep=%d/%d" % (
                case["grid"], case["rows"], case["k"], repetition, args.repetitions)
            if args.dry_run:
                print("[dry-run] %s: %s" % (label, " ".join(command)))
                continue
            try:
                metrics, _ = bencher.execute(command, env=env)
            except Exception as error:
                failures += 1
                bencher.record_failure("e1", command, error)
                print("FAILED %s: %s" % (label, error), file=sys.stderr)
                continue
            row = {
                "experiment": "E1-" + case["grid"],
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "run_id": run_id, "repetition": repetition,
                "implementation": args.implementation,
                "dataset": args.data.name + ("/rows=%d" % case["rows"]),
                "rows": case["rows"], "features": metrics.get("features", 54),
                "k": case["k"], "max_iter": args.max_iter,
                "iterations": metrics["iterations"],
                "stopping_mode": metrics.get("stopping_mode", "fixed_iterations"),
                "converged": metrics.get("converged", "false"),
                "ranks": 1, "threads": args.threads, "total_workers": args.threads,
                "weak": 0, "obs_per_rank": case["rows"],
                "fit_s": float(metrics["fit_s"]),
                "fit_min_s": float(metrics.get("fit_min_s", metrics["fit_s"])),
                "fit_imbalance_s": float(metrics.get("fit_imbalance_s", 0.0)),
                "load_s": float(metrics["load_s"]),
                "preprocess_s": float(metrics["preprocess_s"]),
                "init_s": float(metrics["init_s"]),
                "distribution_s": float(metrics.get("distribution_s", 0.0)),
                "mpi_reduction_s": float(metrics.get("mpi_reduction_s", 0.0)),
                "local_compute_s": float(metrics.get("local_compute_s", 0.0)),
                "thread_reduction_s": float(metrics.get("thread_reduction_s", 0.0)),
                "centroid_update_s": float(metrics.get("centroid_update_s", 0.0)),
                "validation_s": float(metrics["validation_s"]),
                "total_s": float(metrics["total_s"]),
                "sse": float(metrics["inertia"]),
                "label_hash": metrics["label_hash"],
                "notes": "systematic_subsample=%d;threads=%d" % (case["rows"], args.threads),
            }
            row.update(meta)
            bencher.append_row(args.results, bencher.SCHEMA, row)
            print("%s: fit_s=%.4f total_s=%.4f" % (label, row["fit_s"], row["total_s"]))

    if failures:
        print("%d run(s) failed; see %s" % (failures, bencher.FAILURES), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
