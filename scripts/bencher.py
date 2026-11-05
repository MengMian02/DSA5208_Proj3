#!/usr/bin/env python3
"""Shared benchmark harness for the DSA5208 K-means experiments.

Responsibilities kept in one place so every experiment records the same
reproducibility metadata and appends rather than overwrites results:

* environment probing (host, CPU, compiler, MPI, flags)
* running one configuration and parsing the machine-readable key=value output
  of the C++ programs
* appending a structured row to results/<experiment>.csv
* refusing to record a run that failed or produced incomplete metrics
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FAILURES = RESULTS / "failures.log"

# inertia is the program's name for the SSE (sum of squared errors).
REQUIRED_KEYS = ("iterations", "fit_s", "inertia", "label_hash")

SCHEMA = [
    "experiment", "timestamp", "run_id", "repetition", "implementation",
    "host", "cpu_model", "cores", "dataset", "rows", "features", "k",
    "max_iter", "iterations", "stopping_mode", "converged", "ranks", "threads",
    "total_workers", "weak", "obs_per_rank", "fit_s", "fit_min_s",
    "fit_imbalance_s", "load_s", "preprocess_s", "init_s", "distribution_s",
    "mpi_reduction_s", "local_compute_s", "thread_reduction_s",
    "centroid_update_s", "validation_s", "total_s", "sse", "label_hash",
    "compiler", "compiler_flags", "mpi_version", "notes",
]

GPU_SCHEMA = [
    "experiment", "timestamp", "run_id", "repetition", "implementation",
    "stage", "host", "cpu_model", "cores", "device", "dataset", "rows",
    "features", "k", "max_iter", "iterations", "stopping_mode", "converged",
    "fit_s", "sse", "cpu_assign_s", "gpu_assign_s", "gpu_kernel_s",
    "gpu_kernel_total_s", "gpu_host_total_s", "dispatch_floor_s", "upload_s",
    "readback_s", "accumulate_s", "centroid_pack_s", "label_agreement",
    "centroid_max_abs_diff", "inertia_rel_diff", "label_hash",
    "compiler", "compiler_flags", "notes",
]


class BenchmarkError(RuntimeError):
    pass


def _cmd(args, timeout=20):
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (out.stdout or out.stderr).strip()


def cpu_model():
    if sys.platform == "darwin":
        return _cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def mpi_version():
    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    if not launcher:
        return "none"
    text = _cmd([launcher, "--version"])
    for line in text.splitlines():
        if "ersion" in line:
            return line.strip()
    return text.splitlines()[0].strip() if text else "unknown"


def compiler_line(tool):
    path = shutil.which(tool)
    if not path:
        return tool + " (missing)"
    return _cmd([path, "--version"]).splitlines()[0].strip()


def environment(flags=None):
    """Environment metadata attached to every recorded row."""
    return {
        "host": socket.gethostname(),
        "cpu_model": cpu_model(),
        "cores": os.cpu_count() or 0,
        "compiler": compiler_line(os.environ.get("CXX", "g++")) + " | " +
                    compiler_line(os.environ.get("MPICXX", "mpicxx")),
        "compiler_flags": flags or os.environ.get("BENCH_FLAGS", "-O3 -std=c++17 -DNDEBUG"),
        "mpi_version": mpi_version(),
    }


def parse_metrics(stdout):
    """Parse every key=value token from the program output."""
    metrics = {}
    for line in stdout.splitlines():
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            metrics[key] = value
    return metrics


def execute(command, env=None, timeout=3600, cwd=None):
    """Run one configuration, return (metrics, wall seconds). Raises on failure."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    started = _dt.datetime.now()
    process = subprocess.run(command, cwd=str(cwd or ROOT), env=full_env, text=True,
                             capture_output=True, timeout=timeout)
    wall = (_dt.datetime.now() - started).total_seconds()
    if process.returncode != 0:
        raise BenchmarkError(
            "command failed (%d): %s\nstderr tail:\n%s"
            % (process.returncode, shlex.join(command), process.stderr[-1500:])
        )
    metrics = parse_metrics(process.stdout)
    missing = [key for key in REQUIRED_KEYS if key not in metrics]
    if missing:
        raise BenchmarkError(
            "output of %s is missing %s; truncated stdout:\n%s"
            % (shlex.join(command), missing, process.stdout[-1200:])
        )
    metrics["_wall_s"] = wall
    metrics["_stdout"] = process.stdout
    return metrics, wall


def append_row(path, schema, row):
    """Append one row; never truncate an existing results file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    empty = not path.exists() or path.stat().st_size == 0
    if not empty:
        with path.open("r", newline="") as handle:
            header = next(csv.reader(handle), [])
        if header != schema:
            raise BenchmarkError(
                "%s has a different schema; move it aside before recording new results" % path
            )
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=schema, extrasaction="ignore")
        if empty:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in schema})


def record_failure(experiment, command, error):
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    with FAILURES.open("a") as handle:
        handle.write("[%s] %s %s\n  %s\n" % (stamp, experiment, shlex.join(command), error))


def make_run_id(experiment, suffix=""):
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    return experiment + "-" + stamp + (("-" + suffix) if suffix else "")


def require_binary(path):
    if not path.is_file():
        raise BenchmarkError("missing binary %s; run 'make' first" % path)
    return path


def probe():
    import json
    print(json.dumps(environment(), indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--probe":
        raise SystemExit(probe())
    print(__doc__)
    print("Use scripts/run_e1.py, scripts/run_mpi.py or scripts/run_metal.py.")
