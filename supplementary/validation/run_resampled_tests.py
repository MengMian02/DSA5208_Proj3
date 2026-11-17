#!/usr/bin/env python3
"""Fixed-total resampling regression tests.

Verifies that --resampled-total N is rank-count independent and agrees across
serial, OpenMP, MPI and hybrid. Skips cleanly when a binary or MPI launcher is
not available (the Vanda E0 gate runs the same checks with the real launcher).
"""

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/covtype.data"
K = 7
ITERS = 8
SEED = 42
N = 100003  # deliberately not divisible by 2 or 4


def parse(stdout):
    metrics = {}
    for line in stdout.splitlines():
        for token in line.split():
            if "=" in token:
                key, value = token.split("=", 1)
                metrics[key] = value
    return metrics


def run(command, env=None, timeout=300):
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    process = subprocess.run([str(v) for v in command], cwd=str(ROOT), env=full_env,
                             universal_newlines=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, timeout=timeout)
    if process.returncode != 0:
        raise RuntimeError("command failed: %s\n%s" % (" ".join(map(str, command)),
                                                       process.stderr[-1200:]))
    metrics = parse(process.stdout)
    for key in ("rows", "iterations", "inertia", "label_hash"):
        if key not in metrics:
            raise RuntimeError("missing %s in output of %s" % (key, command[0]))
    return metrics


def main():
    required = [ROOT / n for n in ("kmeans_serial", "kmeans_openmp", "kmeans_mpi", "kmeans_hybrid")]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        print("PREREQUISITE MISSING: %s; resampling test skipped" % ", ".join(missing))
        return 0
    if not DATA.is_file():
        raise SystemExit("missing data/covtype.data")

    base = [DATA, K, ITERS, SEED, "--fixed-iterations", "--resampled-total", N]
    serial = run([required[0]] + base)
    reference = serial["label_hash"]
    print("serial N=%s rows=%s hash=%s inertia=%s"
          % (serial["rows"], N, reference, serial["inertia"]))

    checks = [
        ("openmp 4t", [required[1]] + base, {"OMP_NUM_THREADS": "4"}),
        ("mpi 1r", [required[2]] + base, {}),
        ("mpi 2r", [required[2]] + base, {}),
        ("mpi 4r", [required[2]] + base, {}),
        ("hybrid 2x2", [required[3]] + base, {"OMP_NUM_THREADS": "2"}),
    ]
    launcher = shutil.which(os.environ.get("MPI_LAUNCHER", "mpirun")) or shutil.which("mpiexec")
    failures = 0
    for label, command, env in checks:
        if ("mpi" in label or "hybrid" in label):
            if not launcher:
                print("  %-12s SKIP (no MPI launcher)" % label)
                continue
            if label.endswith("1r"):
                command = [launcher, "-np", "1"] + command
            elif label.endswith("2r"):
                command = [launcher, "-np", "2"] + command
            elif label.endswith("4r"):
                command = [launcher, "-np", "4"] + command
            else:
                command = [launcher, "-np", "2"] + command
        result = run(command, env=env)
        same_hash = result["label_hash"] == reference
        same_inertia = math.isclose(float(result["inertia"]), float(serial["inertia"]),
                                    rel_tol=1e-9, abs_tol=1e-4)
        ok = same_hash and same_inertia and result["rows"] == str(N)
        print("  %-12s rows=%s hash=%s inertia=%s %s"
              % (label, result["rows"], result["label_hash"], result["inertia"],
                 "OK" if ok else "MISMATCH"))
        if not ok:
            failures += 1

    if failures:
        raise SystemExit("%d resampling check(s) failed" % failures)
    print("PASS: fixed-total resampling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
