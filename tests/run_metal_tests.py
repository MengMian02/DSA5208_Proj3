#!/usr/bin/env python3
"""Correctness tests for the Apple Metal implementation (local only).

Skipping is explicit: on a machine without Metal the suite reports SKIP and
exits 0, so the CPU test suites stay usable on a Linux cluster.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METAL = ROOT / "kmeans_metal"
SERIAL = ROOT / "kmeans_serial"
DATA = ROOT / "data/covtype.data"


class Failure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Failure(message)


def parse_metrics(stdout):
    metrics = {}
    for line in stdout.splitlines():
        for token in line.split():
            if "=" in token:
                key, value = token.split("=", 1)
                metrics[key] = value
    return metrics


def run(command, timeout=900):
    process = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True,
                             timeout=timeout, env=dict(os.environ))
    return process


def main():
    if sys.platform != "darwin":
        print("SKIP: the Metal implementation is macOS only")
        return 0
    if not METAL.is_file():
        print("SKIP: kmeans_metal not built (make gpu)")
        return 0

    probe = run([str(METAL), str(DATA), "7", "1", "42", "--rows", "500", "--fixed-iterations"])
    if probe.returncode != 0:
        if "Metal unavailable" in probe.stderr:
            print("SKIP: no Metal device on this machine (%s)" % probe.stderr.strip().splitlines()[-1])
            return 0
        print("FAIL: metal probe failed:\n%s" % probe.stderr[-1200:], file=sys.stderr)
        return 1

    temp = Path(tempfile.mkdtemp(prefix="kmeans-metal-tests-"))
    try:
        print("1. Stage A and Stage B agree with each other")
        results = {}
        for mode in ("assign", "full"):
            out = run([str(METAL), str(DATA), "7", "8", "42", "--rows", "5000",
                       "--fixed-iterations", "--mode", mode])
            require(out.returncode == 0, "mode %s failed:\n%s" % (mode, out.stderr[-800:]))
            results[mode] = parse_metrics(out.stdout)
        for key in ("label_hash", "inertia", "cluster_sizes"):
            require(results["assign"][key] == results["full"][key],
                    "stage A and B disagree on %s" % key)
        print("   identical label_hash=%s" % results["full"]["label_hash"])

        print("2. GPU result matches the double-precision CPU reference")
        for rows in (5000, 100000):
            gpu = parse_metrics(run([str(METAL), str(DATA), "7", "10", "42",
                                     "--rows", str(rows), "--fixed-iterations",
                                     "--mode", "full"]).stdout)
            cpu = parse_metrics(run([str(SERIAL), str(DATA), "7", "10", "42",
                                     "--rows", str(rows), "--fixed-iterations"]).stdout)
            agreement = float(gpu["label_agreement"])
            require(agreement >= 0.999,
                    "GPU/CPU label agreement %.6f below 0.999 at N=%d" % (agreement, rows))
            gpu_inertia = float(gpu["inertia"])
            cpu_inertia = float(cpu["inertia"])
            relative = abs(gpu_inertia - cpu_inertia) / cpu_inertia
            require(relative < 1e-4,
                    "inertia differs by %.3e relative at N=%d" % (relative, rows))
            require(int(gpu["cluster_sizes"].split(",")[0]) != 0 or float(cpu["inertia"]) > 0,
                    "unexpected empty clustering")
            print("   N=%d agreement=%.6f inertia rel.diff=%.3e (float GPU vs double CPU)"
                  % (rows, agreement, relative))

        print("3. K = 1 assigns every observation to cluster 0")
        k1 = parse_metrics(run([str(METAL), str(DATA), "1", "5", "42", "--rows", "2000",
                                "--fixed-iterations", "--mode", "full"]).stdout)
        sizes = [int(value) for value in k1["cluster_sizes"].split(",")]
        require(sizes == [2000], "K=1 cluster sizes are %s" % sizes)
        cpu_k1 = parse_metrics(run([str(SERIAL), str(DATA), "1", "5", "42", "--rows", "2000",
                                    "--fixed-iterations"]).stdout)
        require(k1["label_hash"] == cpu_k1["label_hash"], "K=1 label hash differs from CPU")
        print("   cluster_sizes=%s, label_hash matches serial" % sizes)

        print("4. convergence-mode run reaches the same stopping decision as the CPU")
        gpu = parse_metrics(run([str(METAL), str(DATA), "7", "100", "42", "--rows", "20000",
                                 "--mode", "full"]).stdout)
        cpu = parse_metrics(run([str(SERIAL), str(DATA), "7", "100", "42", "--rows", "20000"]).stdout)
        require(gpu["converged"] == cpu["converged"],
                "converged flag differs: gpu=%s cpu=%s" % (gpu["converged"], cpu["converged"]))
        print("   iterations gpu=%s cpu=%s, converged=%s"
              % (gpu["iterations"], cpu["iterations"], gpu["converged"]))

        print("5. repeated GPU runs are bitwise identical")
        first = parse_metrics(run([str(METAL), str(DATA), "7", "6", "42", "--rows", "3000",
                                   "--fixed-iterations"]).stdout)
        second = parse_metrics(run([str(METAL), str(DATA), "7", "6", "42", "--rows", "3000",
                                    "--fixed-iterations"]).stdout)
        require(first["label_hash"] == second["label_hash"], "GPU run is not deterministic")
        print("   identical label_hash=%s" % first["label_hash"])

        print("PASS: Metal assignment and GPU-assisted iteration validated")
        return 0
    except Failure as error:
        print("FAIL: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
