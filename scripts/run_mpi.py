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
    """Launcher mapping/binding flags for one case.

    Vanda's OpenMPI 4.1.8 + PBS TM launch was observed to hang for every
    mapping form that used pe=<threads> (ppr:N:node:pe=T, slot:pe=T, node:pe=T).
    Production hybrid runs therefore use ppr:R:node with --bind-to none and let
    the OS spread the OpenMP team over the exclusive allocation. The old "ppr"
    mode is kept only for diagnostics and is not used by production scripts.
    """
    if args.binding == "ppr-none":
        nodes = max(1, args.nodes)
        per_node = max(1, (case["ranks"] + nodes - 1) // nodes)
        return ["--map-by", "ppr:%d:node" % per_node, "--bind-to", "none"]
    if args.binding == "none":
        return ["--bind-to", "none"]
    if args.experiment == "hybrid" and args.binding == "ppr":
        nodes = max(1, args.nodes)
        per_node = max(1, (case["ranks"] + nodes - 1) // nodes)
        return ["--map-by", "ppr:%d:node:pe=%d" % (per_node, case["threads"]),
                "--bind-to", "core"]
    return mpi_flags()


def build_command(args, case, binary, preflight=False):
    command = [launcher()]
    if args.hostfile:
        command += ["--hostfile", str(args.hostfile)]
    command += case_flags(args, case)
    if args.oversubscribe:
        command.append("--oversubscribe")
    max_iter = 1 if preflight else args.max_iter
    command += [rank_flag(), str(case["ranks"]), str(binary), str(args.data)]
    command += [str(args.k), str(max_iter), str(SEED), "--fixed-iterations"]
    if preflight:
        command += ["--rows", str(args.preflight_rows)]
    else:
        if args.rows:
            command += ["--rows", str(args.rows)]
        if args.resampled_total:
            command += ["--resampled-total", str(args.resampled_total)]
    if case.get("weak_per_rank"):
        command += ["--weak-per-rank", str(case["weak_per_rank"])]
    if args.source_bin:
        command += ["--source-bin", str(args.source_bin)]
    return command


def bounded_execute(command, env, timeout_s):
    """Run a command with an OS-level timeout when available.

    subprocess.run(timeout=...) kills only the direct child; the timeout(1)
    wrapper also gives mpirun a chance to tear down orted children. A timeout
    is a failed configuration, never a benchmark row.
    """
    wrapped = command
    timeout_bin = shutil.which("timeout")
    if timeout_s and timeout_bin:
        wrapped = [timeout_bin, "-k", "10", str(timeout_s)] + command
    return bencher.execute(wrapped, env=env, timeout=(timeout_s + 30) if timeout_s else 3600)


def check_worker_count(metrics, case, label):
    expected = case["ranks"] * case["threads"]
    got_ranks = int(metrics.get("ranks", -1))
    got_threads = int(metrics.get("threads", -1))
    got_workers = int(metrics.get("total_workers", -1))
    if got_ranks != case["ranks"] or got_threads != case["threads"] or got_workers != expected:
        raise bencher.BenchmarkError(
            "%s: reported ranks/threads/workers %s/%s/%s, expected %s/%s/%s"
            % (label, got_ranks, got_threads, got_workers,
               case["ranks"], case["threads"], expected))


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
    parser.add_argument("--resampled-total", type=int, default=0,
                        help="fixed logical dataset size generated by deterministic "
                             "resampling (same N for every rank count)")
    parser.add_argument("--dataset-label", type=str, default="",
                        help="value recorded in the dataset column (default: data file name)")
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
    parser.add_argument("--binding", choices=["core", "ppr", "ppr-none", "none"],
                        default="core",
                        help="core: env MPI_FLAGS; ppr-none: ppr:R:node + bind none "
                             "(validated hybrid path); ppr: legacy pe=T probe; "
                             "none: bind none only")
    parser.add_argument("--preflight-rows", type=int, default=0,
                        help="hybrid only: run one short launch/placement check per "
                             "layout before the timed repetitions (0 = off)")
    parser.add_argument("--preflight-timeout", type=int, default=90,
                        help="seconds allowed for one layout preflight")
    parser.add_argument("--timeout", type=int, default=600,
                        help="seconds allowed for one timed run (OS timeout wrapper)")
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
    cases = build_cases(args)
    for case in cases:
        if args.experiment == "hybrid" and args.preflight_rows > 0 and not args.dry_run:
            check_command = build_command(args, case, binary, preflight=True)
            check_label = "%s preflight ranks=%d threads=%d" % (
                experiment, case["ranks"], case["threads"])
            print("PREFLIGHT %s: %s" % (check_label, " ".join(check_command)))
            try:
                metrics, _ = bounded_execute(
                    check_command,
                    {"OMP_NUM_THREADS": str(case["threads"])},
                    args.preflight_timeout)
                check_worker_count(metrics, case, check_label)
                print("PREFLIGHT PASS %s hash=%s"
                      % (check_label, metrics.get("label_hash", "?")))
            except Exception as error:
                failures += 1
                bencher.record_failure("e" + experiment[-1], check_command, error)
                print("PREFLIGHT FAILED %s: %s" % (check_label, error), file=sys.stderr)
                continue
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
                metrics, _ = bounded_execute(command, env, args.timeout)
                check_worker_count(metrics, case, label)
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
                "dataset": args.dataset_label or args.data.name,
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
                "notes": "weak_per_rank=%s;resampled_total=%s;source_bin=%s;"
                         "binding=%s;nodes=%s" % (
                    case.get("weak_per_rank", 0), args.resampled_total,
                    bool(args.source_bin), args.binding, args.nodes),
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
