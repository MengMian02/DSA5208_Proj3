#!/usr/bin/env python3
"""E6 - local Apple Metal experiment (assignment kernel and full iteration).

Stage A (--mode assign) re-uploads the observation matrix and the centroids on
every iteration and copies labels back, i.e. the naive transfer pattern.
Stage B (--mode full) keeps the observation matrix resident and only re-uploads
the K x D centroids.

Both stages are validated in-run against a double-precision CPU reference with
the same initialization and iteration count; the CPU/GPU comparison is
tolerance based because Metal has no IEEE double.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bencher

SEED = 42


def main():
    parser = argparse.ArgumentParser(description="E6 Metal benchmark")
    parser.add_argument("--data", type=Path, default=bencher.ROOT / "data/covtype.data")
    parser.add_argument("--rows", type=str, default="20000,50000,100000,200000")
    parser.add_argument("--modes", type=str, default="assign,full")
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--results", type=Path, default=bencher.RESULTS / "e6_metal.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    binary = bencher.require_binary(bencher.ROOT / "kmeans_metal")
    meta = bencher.environment()
    run_id = bencher.make_run_id("e6")
    rows_list = [int(value) for value in args.rows.split(",") if value.strip()]
    modes = [value for value in args.modes.split(",") if value.strip()]

    failures = 0
    for mode in modes:
        for rows in rows_list:
            for repetition in range(1, args.repetitions + 1):
                command = [str(binary), str(args.data), str(args.k), str(args.max_iter),
                           str(SEED), "--fixed-iterations", "--rows", str(rows),
                           "--mode", mode, "--bench-overhead"]
                label = "mode=%s rows=%d rep=%d/%d" % (mode, rows, repetition, args.repetitions)
                if args.dry_run:
                    print("[dry-run] %s: %s" % (label, " ".join(command)))
                    continue
                try:
                    metrics, _ = bencher.execute(command)
                except Exception as error:
                    failures += 1
                    bencher.record_failure("e6", command, error)
                    print("FAILED %s: %s" % (label, error), file=sys.stderr)
                    continue
                row = {
                    "experiment": "E6",
                    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                    "run_id": run_id, "repetition": repetition,
                    "implementation": "metal",
                    "stage": metrics.get("stage", mode),
                    "device": metrics.get("device", "unknown"),
                    "dataset": args.data.name + ("/rows=%d" % rows),
                    "rows": rows, "features": metrics.get("features", 54),
                    "k": args.k, "max_iter": args.max_iter,
                    "iterations": metrics["iterations"],
                    "stopping_mode": metrics.get("stopping_mode", "fixed_iterations"),
                    "converged": metrics.get("converged", "false"),
                    "fit_s": float(metrics["fit_s"]),
                    "sse": float(metrics["inertia"]),
                    "cpu_assign_s": float(metrics.get("cpu_assign_s", 0.0)),
                    "gpu_assign_s": float(metrics.get("gpu_assign_s", 0.0)),
                    "gpu_kernel_s": float(metrics.get("gpu_kernel_s", 0.0)),
                    "gpu_kernel_total_s": float(metrics.get("gpu_kernel_total_s", 0.0)),
                    "gpu_host_total_s": float(metrics.get("gpu_host_total_s", 0.0)),
                    "dispatch_floor_s": float(metrics.get("dispatch_floor_s", 0.0)),
                    "upload_s": float(metrics.get("upload_s", 0.0)),
                    "readback_s": float(metrics.get("readback_s", 0.0)),
                    "accumulate_s": float(metrics.get("accumulate_s", 0.0)),
                    "centroid_pack_s": float(metrics.get("centroid_pack_s", 0.0)),
                    "label_agreement": float(metrics.get("label_agreement", 0.0)),
                    "centroid_max_abs_diff": float(metrics.get("centroid_max_abs_diff", 0.0)),
                    "inertia_rel_diff": float(metrics.get("inertia_rel_diff", 0.0)),
                    "label_hash": metrics["label_hash"],
                    "notes": "mode=%s;single_precision_gpu" % mode,
                }
                row.update(meta)
                bencher.append_row(args.results, bencher.GPU_SCHEMA, row)
                print("%s: fit_s=%.4f cpu_assign=%.5f gpu_kernel=%.5f agreement=%.6f"
                      % (label, row["fit_s"], row["cpu_assign_s"], row["gpu_kernel_s"],
                         row["label_agreement"]))

    if failures:
        print("%d run(s) failed; see %s" % (failures, bencher.FAILURES), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
