#!/usr/bin/env python3
"""E7 - CUDA K-means assignment experiment on NUS Vanda (NVIDIA A40).

Mirrors scripts/run_metal.py so the two GPU platforms produce the same columns:
Stage A re-uploads the observation matrix every iteration, Stage B keeps it
resident. Every run is validated in-process against a double-precision CPU
reference with the same initialization and iteration count.
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bencher

SEED = 42


def main():
    parser = argparse.ArgumentParser(description="E7 CUDA benchmark")
    parser.add_argument("--data", type=Path, default=bencher.ROOT / "data/covtype.data")
    parser.add_argument("--rows", type=str, default="50000,100000,200000,400000,581012")
    parser.add_argument("--modes", type=str, default="assign,full")
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--results", type=Path, default=bencher.RESULTS / "e7_cuda.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    binary = bencher.require_binary(bencher.ROOT / "kmeans_cuda")
    meta = bencher.environment()
    run_id = bencher.make_run_id("e7")
    rows_list = [int(v) for v in args.rows.split(",") if v.strip()]
    modes = [v for v in args.modes.split(",") if v.strip()]

    failures = 0
    for mode in modes:
        for rows in rows_list:
            for repetition in range(1, args.repetitions + 1):
                command = [str(binary), str(args.data), str(args.k), str(args.max_iter),
                           str(SEED), "--fixed-iterations", "--rows", str(rows),
                           "--mode", mode]
                label = "mode=%s rows=%d rep=%d/%d" % (mode, rows, repetition, args.repetitions)
                if args.dry_run:
                    print("[dry-run] %s: %s" % (label, " ".join(command)))
                    continue
                try:
                    metrics, _ = bencher.execute(command)
                except Exception as error:
                    failures += 1
                    bencher.record_failure("e7", command, error)
                    print("FAILED %s: %s" % (label, error), file=sys.stderr)
                    continue

                def num(key, default=0.0):
                    value = metrics.get(key)
                    return float(value) if value not in (None, "") else default

                row = {
                    "experiment": "E7",
                    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                    "run_id": run_id, "repetition": repetition,
                    "implementation": "cuda",
                    "stage": metrics.get("stage", mode),
                    "device": metrics.get("device", "unknown"),
                    "dataset": args.data.name + ("/rows=%d" % rows),
                    "rows": rows, "features": metrics.get("features", 54),
                    "k": args.k, "max_iter": args.max_iter,
                    "iterations": metrics["iterations"],
                    "stopping_mode": metrics.get("stopping_mode", "fixed_iterations"),
                    "converged": metrics.get("converged", "false"),
                    "fit_s": num("fit_s"),
                    "sse": num("inertia"),
                    "cpu_assign_s": num("cpu_assign_s"),
                    "gpu_assign_s": num("gpu_assign_s"),
                    "gpu_kernel_s": num("gpu_kernel_s"),
                    "dispatch_floor_s": num("dispatch_floor_s"),
                    "upload_s": num("upload_s"),
                    "readback_s": num("readback_s"),
                    "accumulate_s": num("accumulate_s"),
                    "centroid_pack_s": num("centroid_pack_s"),
                    "label_agreement": num("label_agreement"),
                    "centroid_max_abs_diff": num("centroid_max_abs_diff"),
                    "inertia_rel_diff": num("inertia_rel_diff"),
                    "label_hash": metrics["label_hash"],
                    "notes": "mode=%s;single_precision_gpu" % mode,
                }
                row.update(meta)
                bencher.append_row(args.results, bencher.GPU_SCHEMA, row)
                print("%s: fit_s=%.4f cpu_assign=%.5f gpu_kernel=%.6f agree=%.6f rel=%.2e"
                      % (label, row["fit_s"], row["cpu_assign_s"], row["gpu_kernel_s"],
                         row["label_agreement"], row["inertia_rel_diff"]))

    if failures:
        print("%d run(s) failed; see %s" % (failures, bencher.FAILURES), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
