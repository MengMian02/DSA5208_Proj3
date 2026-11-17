#!/usr/bin/env python3
"""E5 - OpenMP-only thread scaling on one exclusive CPU node.

The PBS allocation is always 64 cores; OMP_NUM_THREADS is varied per run.
Rows carry the actual thread count reported by the binary, so an accidental
OMP_THREAD_LIMIT or dynamic-team change is visible in the CSV.
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bencher

SEED = 42


def main():
    parser = argparse.ArgumentParser(description="E5 OpenMP thread scaling")
    parser.add_argument("--threads", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--data", type=Path, default=bencher.ROOT / "data/covtype.data")
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--rows", type=int, default=0,
                        help="original-data systematic subset (0 = full Covertype)")
    parser.add_argument("--resampled-total", type=int, default=0,
                        help="fixed-total resampled workload (0 = original/rows mode)")
    parser.add_argument("--dataset-label", type=str, default="covtype_original")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    binary = bencher.require_binary(bencher.ROOT / "kmeans_openmp")
    meta = bencher.environment()
    run_id = bencher.make_run_id("e5")
    failures = 0

    for threads in [int(v) for v in args.threads.split(",") if v.strip()]:
        for repetition in range(1, args.repetitions + 1):
            command = [str(binary), str(args.data), str(args.k), str(args.max_iter),
                       str(SEED), "--fixed-iterations"]
            if args.resampled_total:
                command += ["--resampled-total", str(args.resampled_total)]
            elif args.rows:
                command += ["--rows", str(args.rows)]
            env = {"OMP_NUM_THREADS": str(threads)}
            label = "E5 threads=%d rep=%d/%d" % (threads, repetition, args.repetitions)
            if args.dry_run:
                print("[dry-run] %s: OMP_NUM_THREADS=%d %s"
                      % (label, threads, " ".join(command)))
                continue
            try:
                metrics, _ = bencher.execute(command, env=env, timeout=args.timeout)
                reported = int(metrics.get("threads", -1))
                if reported != threads:
                    raise bencher.BenchmarkError(
                        "%s: binary reported %d threads, requested %d"
                        % (label, reported, threads))
            except Exception as error:
                failures += 1
                bencher.record_failure("e5", command, error)
                print("FAILED %s: %s" % (label, error), file=sys.stderr)
                continue

            rows = int(metrics["rows"])
            row = {
                "experiment": "E5",
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "run_id": run_id, "repetition": repetition,
                "implementation": "openmp",
                "dataset": args.dataset_label,
                "rows": rows, "features": metrics.get("features", 54),
                "k": args.k, "max_iter": args.max_iter,
                "iterations": metrics["iterations"],
                "stopping_mode": metrics.get("stopping_mode", "fixed_iterations"),
                "converged": metrics.get("converged", "false"),
                "ranks": 1, "threads": threads, "total_workers": threads,
                "weak": 0, "obs_per_rank": rows,
                "fit_s": float(metrics["fit_s"]),
                "fit_min_s": float(metrics.get("fit_min_s", metrics["fit_s"])),
                "fit_imbalance_s": 0.0,
                "load_s": float(metrics.get("load_s", 0.0)),
                "preprocess_s": float(metrics.get("preprocess_s", 0.0)),
                "init_s": float(metrics.get("init_s", 0.0)),
                "distribution_s": float(metrics.get("distribution_s", 0.0)),
                "mpi_reduction_s": 0.0,
                "local_compute_s": float(metrics.get("local_compute_s", 0.0)),
                "thread_reduction_s": float(metrics.get("thread_reduction_s", 0.0)),
                "centroid_update_s": float(metrics.get("centroid_update_s", 0.0)),
                "validation_s": float(metrics.get("validation_s", 0.0)),
                "total_s": float(metrics.get("total_s", 0.0)),
                "sse": float(metrics["inertia"]),
                "label_hash": metrics["label_hash"],
                "notes": "threads=%d;resampled_total=%d;rows=%d" % (
                    threads, args.resampled_total, args.rows),
            }
            row.update(meta)
            bencher.append_row(args.results, bencher.SCHEMA, row)
            print("%s: fit_s=%.4f total_s=%.4f" % (label, row["fit_s"], row["total_s"]))

    if failures:
        print("%d run(s) failed; see %s" % (failures, bencher.FAILURES), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
