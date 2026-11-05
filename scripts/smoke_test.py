#!/usr/bin/env python3
"""Fast end-to-end smoke test: build check plus cross-implementation agreement.

Runs in well under a minute on the Covertype data and is the first thing to run
after a checkout or a rebuild:

    make && python3 scripts/smoke_test.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/covtype.data"
ROWS = 20000
K = 7
ITERS = 10
SEED = 42


def parse(stdout):
    metrics = {}
    for line in stdout.splitlines():
        for token in line.split():
            if "=" in token:
                key, value = token.split("=", 1)
                metrics[key] = value
    return metrics


def run(command, env=None, expect="label_hash"):
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    process = subprocess.run([str(part) for part in command], cwd=str(ROOT), env=full_env,
                             text=True, capture_output=True, timeout=900)
    if process.returncode != 0:
        raise SystemExit("FAILED: %s\n%s" % (" ".join(map(str, command)), process.stderr[-1500:]))
    metrics = parse(process.stdout)
    if expect not in metrics:
        raise SystemExit("missing %s in output of %s" % (expect, command[0]))
    return metrics


def main():
    missing = [name for name in ("kmeans_serial", "kmeans_openmp", "kmeans_mpi",
                                 "kmeans_hybrid")
               if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("missing binaries: %s (run make)" % ", ".join(missing))
    if not DATA.is_file():
        raise SystemExit("missing dataset data/covtype.data")

    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    common = [str(DATA), str(K), str(ITERS), str(SEED), "--fixed-iterations",
              "--rows", str(ROWS)]

    print("serial reference ...")
    reference = run([ROOT / "kmeans_serial"] + common)
    expected = reference["label_hash"]
    print("  rows=%s iterations=%s inertia=%s hash=%s"
          % (reference["rows"], reference["iterations"], reference["inertia"], expected))

    checks = [("openmp 1x2", [ROOT / "kmeans_openmp"] + common, {"OMP_NUM_THREADS": "2"})]
    if launcher:
        checks += [
            ("mpi 2x1", [launcher, "-np", "2", ROOT / "kmeans_mpi"] + common, {}),
            ("hybrid 2x2", [launcher, "-np", "2", ROOT / "kmeans_hybrid"] + common,
             {"OMP_NUM_THREADS": "2"}),
            ("weak 2 ranks", [launcher, "-np", "2", ROOT / "kmeans_mpi", str(DATA),
                              str(K), str(ITERS), str(SEED), "--fixed-iterations",
                              "--weak-per-rank", "5000"], {}),
        ]
    else:
        print("  (no mpirun: MPI checks skipped)")

    failures = 0
    for label, command, env in checks:
        metrics = run(command, env=env)
        if "weak" in label:
            ok = metrics.get("weak") == "1" and metrics.get("obs_per_rank") == "5000"
            print("  %-14s rows=%s weak=%s obs/rank=%s" %
                  (label, metrics["rows"], metrics["weak"], metrics["obs_per_rank"]))
        else:
            ok = metrics["label_hash"] == expected and metrics["inertia"] == reference["inertia"]
            print("  %-14s hash=%s %s" % (label, metrics["label_hash"], "OK" if ok else "MISMATCH"))
        if not ok:
            failures += 1

    if (ROOT / "kmeans_metal").is_file() and sys.platform == "darwin":
        metrics = run([ROOT / "kmeans_metal"] + common + ["--mode", "full"])
        agreement = float(metrics["label_agreement"])
        print("  metal full     agreement=%.6f inertia=%s" % (agreement, metrics["inertia"]))
        if agreement < 0.999:
            failures += 1
    else:
        print("  (kmeans_metal not built or not macOS: GPU check skipped)")

    if failures:
        raise SystemExit("%d check(s) failed" % failures)
    print("PASS: smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
