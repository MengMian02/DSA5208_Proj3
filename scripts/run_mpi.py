#!/usr/bin/env python3
"""E2/E3/E4/E5 - MPI strong scaling, MPI weak scaling and hybrid MPI+OpenMP.

  strong  fixed dataset, ranks 1..256; rank 1 is the primary MPI baseline
  weak    --weak-per-rank observations per rank, generated in memory from the
          real Covertype source by a deterministic global-index hash
  hybrid  cross product of ranks x OpenMP threads at (optionally) equal core
          budgets

One row per repetition is appended; medians across repetitions feed the plots.
"""


import argparse
import datetime as dt
import os
import shlex
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bencher

SEED = 42

# One file per experiment; the analysis script reads exactly these names.
DEFAULT_RESULTS = {
    "strong": "e2_strong.csv",
    "weak": "e3_weak.csv",
    "hybrid": "e4_hybrid.csv",
}


def parse_int_list(text):
    return [int(value) for value in text.split(",") if value.strip()]


def parse_rank_list(text):
    return [int(value) for value in text.split(",") if value.strip()]


def launcher():
    """Site launcher: MPI_LAUNCHER when set, otherwise mpirun/mpiexec."""
    configured = os.environ.get("MPI_LAUNCHER", "").strip()
    path = configured or shutil.which("mpirun") or shutil.which("mpiexec")
    if not path:
        raise bencher.BenchmarkError("no MPI launcher found (set MPI_LAUNCHER)")
    return path


def mpi_flags():
    """Extra launcher flags, e.g. binding options, from MPI_FLAGS."""
    return shlex.split(os.environ.get("MPI_FLAGS", ""))


def rank_flag():
    return "-n" if os.path.basename(launcher()) == "srun" else "-np"


def build_cases(args):
    cases = []
    if args.experiment == "strong":
        for ranks in args.ranks:
            cases.append({"implementation": "mpi", "ranks": ranks, "threads": 1})
    elif args.experiment == "weak":
        for ranks in args.ranks:
            cases.append({"implementation": "mpi", "ranks": ranks, "threads": 1,
                          "weak_per_rank": args.per_rank})
    elif args.layouts:  # explicit ranks x threads layouts
        for layout in args.layouts:
            ranks_text, _, threads_text = layout.lower().partition("x")
            if not ranks_text or not threads_text:
                raise bencher.BenchmarkError("bad layout %r, expected RANKSxTHREADS" % layout)
            cases.append({"implementation": "hybrid", "ranks": int(ranks_text),
                          "threads": int(threads_text)})
    else:  # hybrid cross product
        for ranks in args.ranks:
            for threads in args.threads:
                workers = ranks * threads
                if args.max_workers and workers > args.max_workers:
                    continue
                cases.append({"implementation": "hybrid", "ranks": ranks, "threads": threads})
    return cases


def case_flags(args, case):
    """Launcher binding flags for one case.

    Hybrid layouts at an equal core budget need one rank block of T cores per
    rank (ppr:...:pe=T), otherwise --bind-to core would pin a whole OpenMP team
    onto a single core. Single-threaded sweeps keep one rank per core.
    """
    if args.binding == "none":
        return ["--bind-to", "none"]
    if args.experiment == "hybrid" and args.binding == "ppr":
        nodes = max(1, args.nodes)
        per_node = max(1, (case["ranks"] + nodes - 1) // nodes)
        return ["--map-by", "ppr:%d:node:pe=%d" % (per_node, case["threads"]),
                "--bind-to", "core"]
    return mpi_flags()


def build_command(args, case, binary):
    command = [launcher()]
    if args.hostfile:
        command += ["--hostfile", str(args.hostfile)]
    command += case_flags(args, case)
    if args.oversubscribe:
        command.append("--oversubscribe")
    command += [rank_flag(), str(case["ranks"]), str(binary), str(args.data)]
    command += [str(args.k), str(args.max_iter), str(SEED), "--fixed-iterations"]
    if args.rows:
        command += ["--rows", str(args.rows)]
    if case.get("weak_per_rank"):
        command += ["--weak-per-rank", str(case["weak_per_rank"])]
    if args.source_bin:
        command += ["--source-bin", str(args.source_bin)]
    return command


def main():
    parser = argparse.ArgumentParser(description="MPI scaling benchmark driver")
    parser.add_argument("--experiment", choices=["strong", "weak", "hybrid"], required=True)
    parser.add_argument("--data", type=Path, default=bencher.ROOT / "data/covtype.data")
    parser.add_argument("--ranks", type=str, default="1,2,4,8")
    parser.add_argument("--threads", type=str, default="1,2,4")
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--rows", type=int, default=0,
                        help="systematic subsample size (0 = full Covertype)")
    parser.add_argument("--per-rank", type=int, default=50_000,
                        help="E3: observations per rank")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--layouts", type=str, default="",
                        help="explicit hybrid layouts such as 96x1,48x2,1x96")
    parser.add_argument("--max-workers", type=int, default=0,
                        help="skip layouts above this many MPI ranks x threads")
    parser.add_argument("--source-bin", type=Path, default=None,
                        help="E3: shared cached binary source instead of a broadcast")
    parser.add_argument("--hostfile", type=str, default="",
                        help="explicit launcher hostfile (e.g. the PBS nodefile for a "
                             "multi-node job; Open MPI otherwise sees one node)")
    parser.add_argument("--nodes", type=int, default=1,
                        help="nodes the allocation spans (drives ppr binding)")
    parser.add_argument("--binding", choices=["core", "ppr", "none"], default="core",
                        help="core: env MPI_FLAGS; ppr: one rank block of T cores per "
                             "layout; none: no binding (OpenMP spreads itself)")
    parser.add_argument("--oversubscribe", action="store_true")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.ranks = parse_rank_list(args.ranks)
    args.threads = parse_int_list(args.threads)
    args.layouts = [value for value in args.layouts.split(",") if value.strip()]
    results = args.results or (bencher.RESULTS / DEFAULT_RESULTS[args.experiment])
    binary = bencher.require_binary(bencher.ROOT / "kmeans_mpi")
    if args.experiment == "hybrid":
        binary = bencher.require_binary(bencher.ROOT / "kmeans_hybrid")
    meta = bencher.environment()
    run_id = bencher.make_run_id("e" + {"strong": "2", "weak": "3", "hybrid": "4"}[args.experiment])
    experiment = {"strong": "E2", "weak": "E3", "hybrid": "E4"}[args.experiment]

    failures = 0
    for case in build_cases(args):
        for repetition in range(1, args.repetitions + 1):
            command = build_command(args, case, binary)
            env = {"OMP_NUM_THREADS": str(case["threads"])}
            label = "%s ranks=%d threads=%d rep=%d/%d" % (
                experiment, case["ranks"], case["threads"], repetition, args.repetitions)
            if args.dry_run:
                print("[dry-run] %s: OMP_NUM_THREADS=%d %s"
                      % (label, case["threads"], " ".join(command)))
                continue
            try:
                metrics, _ = bencher.execute(command, env=env)
            except Exception as error:
                failures += 1
                bencher.record_failure("e" + experiment[-1], command, error)
                print("FAILED %s: %s" % (label, error), file=sys.stderr)
                continue
            rows = int(metrics["rows"])
            row = {
                "experiment": experiment,
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "run_id": run_id, "repetition": repetition,
                "implementation": case["implementation"],
                "dataset": args.data.name,
                "rows": rows, "features": metrics.get("features", 54),
                "k": args.k, "max_iter": args.max_iter,
                "iterations": metrics["iterations"],
                "stopping_mode": metrics.get("stopping_mode", "fixed_iterations"),
                "converged": metrics.get("converged", "false"),
                "ranks": int(metrics.get("ranks", case["ranks"])),
                "threads": int(metrics.get("threads", case["threads"])),
                "total_workers": int(metrics.get("total_workers", case["ranks"] * case["threads"])),
                "weak": int(metrics.get("weak", 1 if case.get("weak_per_rank") else 0)),
                "obs_per_rank": int(metrics.get("obs_per_rank", rows // max(case["ranks"], 1))),
                "fit_s": float(metrics["fit_s"]),
                "fit_min_s": float(metrics.get("fit_min_s", metrics["fit_s"])),
                "fit_imbalance_s": float(metrics.get("fit_imbalance_s", 0.0)),
                "load_s": float(metrics.get("load_s", 0.0)),
                "preprocess_s": float(metrics.get("preprocess_s", 0.0)),
                "init_s": float(metrics.get("init_s", 0.0)),
                "distribution_s": float(metrics.get("distribution_s", 0.0)),
                "mpi_reduction_s": float(metrics.get("mpi_reduction_s", 0.0)),
                "local_compute_s": float(metrics.get("local_compute_s", 0.0)),
                "thread_reduction_s": float(metrics.get("thread_reduction_s", 0.0)),
                "centroid_update_s": float(metrics.get("centroid_update_s", 0.0)),
                "validation_s": float(metrics.get("validation_s", 0.0)),
                "total_s": float(metrics.get("total_s", 0.0)),
                "sse": float(metrics["inertia"]),
                "label_hash": metrics["label_hash"],
                "notes": "weak_per_rank=%s;source_bin=%s" % (
                    case.get("weak_per_rank", 0), bool(args.source_bin)),
            }
            row.update(meta)
            bencher.append_row(results, bencher.SCHEMA, row)
            print("%s: fit_s=%.4f local=%.4f coll=%.4f obs/rank=%d"
                  % (label, row["fit_s"], row["local_compute_s"],
                     row["mpi_reduction_s"], row["obs_per_rank"]))

    if failures:
        print("%d run(s) failed; see %s" % (failures, bencher.FAILURES), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
