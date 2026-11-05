#!/usr/bin/env python3
"""Correctness tests for E3 weak scaling with deterministic local generation.

What is verified:

1. The documented index mapping from a global observation index to a source
   Covertype row is reproduced exactly by an independent Python implementation
   and by the rows the C++ program actually materialised.
2. The logical expanded dataset does not depend on the MPI rank count: the same
   (seed, total observations) gives the same clustering under 1, 2 and 4 ranks.
3. Repeated runs are bitwise identical.
4. Reading a shared cached binary source gives the same result as broadcasting.
5. Reported metadata (rows, obs_per_rank, weak flag) is correct.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPI = shutil.which("mpirun") or shutil.which("mpiexec")
BINARY = ROOT / "kmeans_mpi"
SERIAL = ROOT / "kmeans_serial"
MASK = (1 << 64) - 1


class Failure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Failure(message)


def splitmix64(z):
    z = (z + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def weak_source_index(seed, global_index, source_rows):
    stream = splitmix64(seed ^ 0x5DEECE66D)
    mixed = splitmix64(((global_index * 0x9E3779B97F4A7C15) & MASK) ^ stream)
    return mixed % source_rows


def parse_metrics(stdout):
    metrics = {}
    for line in stdout.splitlines():
        for token in line.split():
            if "=" in token:
                key, value = token.split("=", 1)
                metrics[key] = value
    return metrics


def run(command, env=None, timeout=600):
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    process = subprocess.run(command, cwd=str(ROOT), env=full_env, text=True,
                             capture_output=True, timeout=timeout)
    require(process.returncode == 0,
            "command failed: %s\n%s" % (" ".join(map(str, command)), process.stderr[-1200:]))
    return parse_metrics(process.stdout)


def read_source(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line:
            continue
        values = [float(value) for value in line.split(",")]
        rows.append(values[:54])
    return rows


def preprocess(rows):
    """Same Welford z-scoring as kmeans_common.hpp, applied to the first 10 columns."""
    n = len(rows)
    for j in range(10):
        mean = 0.0
        m2 = 0.0
        for i in range(n):
            value = rows[i][j]
            delta = value - mean
            mean += delta / float(i + 1)
            m2 += delta * (value - mean)
        sd = (m2 / float(n)) ** 0.5
        for i in range(n):
            rows[i][j] = (rows[i][j] - mean) / sd if sd > 0 else 0.0
    return rows


def main():
    parser_seed = 4242
    temp = Path(tempfile.mkdtemp(prefix="kmeans-weak-tests-"))
    try:
        require(BINARY.is_file(), "missing kmeans_mpi; run make first")
        require(MPI is not None, "mpirun/mpiexec not found")

        print("1. documented index mapping matches the materialised rows")
        source_csv = temp / "source.csv"
        all_rows = read_source(ROOT / "data/covtype.data")
        # A small but statistically varied slice keeps the test fast while
        # still exercising real Covertype values.
        stride = len(all_rows) // 2000
        subset = all_rows[::stride][:2000]
        source_csv.write_text("\n".join(",".join("%.17g" % v for v in row) + ",1"
                                       for row in subset) + "\n")
        expected = preprocess([[value for value in row] for row in subset])
        for ranks, per_rank in ((1, 200), (3, 200), (2, 150)):
            dump = temp / ("dump-%dx%d.csv" % (ranks, per_rank))
            run([MPI, "-np", str(ranks), str(BINARY), str(source_csv), "7", "3",
                 str(parser_seed), "--fixed-iterations", "--weak-per-rank", str(per_rank),
                 "--dump-generated", str(dump)])
            got = read_source(dump)
            require(len(got) == ranks * per_rank,
                    "expected %d generated rows, got %d" % (ranks * per_rank, len(got)))
            worst = 0.0
            for index, row in enumerate(got):
                source_index = weak_source_index(parser_seed, index, len(expected))
                reference = expected[source_index]
                for a, b in zip(row, reference):
                    worst = max(worst, abs(a - b))
            require(worst < 1e-12,
                    "%dx%d mapping mismatch, max difference %.3e" % (ranks, per_rank, worst))
            print("   %dx%d: %d rows match the documented mapping (max diff %.1e)"
                  % (ranks, per_rank, len(got), worst))

        print("2. the logical dataset is independent of the MPI rank count")
        reference = None
        for ranks in (1, 2, 4):
            metrics = run([MPI, "-np", str(ranks), str(BINARY), str(source_csv), "7", "8",
                           str(parser_seed), "--fixed-iterations",
                           "--weak-per-rank", str(6000 // ranks)])
            require(int(metrics["rows"]) == 6000,
                    "reported rows=%s, expected 6000" % metrics["rows"])
            require(int(metrics["obs_per_rank"]) == 6000 // ranks,
                    "reported obs_per_rank=%s" % metrics["obs_per_rank"])
            require(metrics["weak"] == "1", "weak flag not set")
            key = (metrics["label_hash"], round(float(metrics["inertia"]), 9),
                   metrics["cluster_sizes"])
            if reference is None:
                reference = key
            else:
                require(key == reference,
                        "%d ranks produced a different clustering: %s vs %s"
                        % (ranks, key, reference))
            print("   ranks=%d rows=%s obs/rank=%s hash=%s inertia=%s"
                  % (ranks, metrics["rows"], metrics["obs_per_rank"],
                     metrics["label_hash"], metrics["inertia"]))

        print("3. repeated runs are bitwise identical")
        first = run([MPI, "-np", "2", str(BINARY), str(source_csv), "7", "5", "9",
                     "--fixed-iterations", "--weak-per-rank", "500"])
        second = run([MPI, "-np", "2", str(BINARY), str(source_csv), "7", "5", "9",
                      "--fixed-iterations", "--weak-per-rank", "500"])
        require(first["label_hash"] == second["label_hash"], "repeated run differs")
        require(first["inertia"] == second["inertia"], "repeated run inertia differs")
        print("   identical label_hash=%s" % first["label_hash"])

        print("4. cached binary source equals the broadcast path")
        binary_path = temp / "source.bin"
        broadcast = run([MPI, "-np", "2", str(BINARY), str(source_csv), "7", "5", "9",
                         "--fixed-iterations", "--weak-per-rank", "700"])
        cached = run([MPI, "-np", "2", str(BINARY), str(source_csv), "7", "5", "9",
                      "--fixed-iterations", "--weak-per-rank", "700",
                      "--source-bin", str(binary_path)])
        require(binary_path.is_file(), "source binary was not written")
        require(broadcast["label_hash"] == cached["label_hash"],
                "cached source changes the solution")
        print("   identical label_hash=%s" % cached["label_hash"])

        print("5. strong-scaling path is unaffected by the weak-scaling code")
        for ranks in (1, 3, 4):
            strong = run([MPI, "-np", str(ranks), str(BINARY), str(source_csv), "7", "6",
                          "42", "--fixed-iterations"])
            serial = run([str(SERIAL), str(source_csv), "7", "6", "42", "--fixed-iterations"])
            require(strong["label_hash"] == serial["label_hash"],
                    "%d-rank strong run differs from serial" % ranks)
        print("   strong mode still matches serial bitwise")
        print("PASS: weak-scaling generation, determinism and reporting verified")
        return 0
    except Failure as error:
        print("FAIL: %s" % error, file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
