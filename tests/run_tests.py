#!/usr/bin/env python3
"""Cross-check serial, OpenMP, MPI, and hybrid K-means implementations."""

import argparse
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BINARIES = {
    "serial": ROOT / "kmeans_serial",
    "openmp": ROOT / "kmeans_openmp",
    "mpi": ROOT / "kmeans_mpi",
    "hybrid": ROOT / "kmeans_hybrid",
}


class TestFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise TestFailure(message)


def parse_metrics(stdout):
    metrics = {}
    for line in stdout.splitlines():
        if not line:
            continue
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            metrics[key] = value
    require("inertia" in metrics, "program output has no inertia")
    for key in ("rows", "k", "max_iter", "iterations", "label_hash"):
        require(key in metrics, f"program output has no {key}")
        metrics[key] = int(metrics[key])
    metrics["inertia"] = float(metrics["inertia"])
    metrics["cluster_sizes"] = [int(value) for value in metrics["cluster_sizes"].split(",")]
    metrics["converged"] = metrics["converged"] == "true"
    return metrics


def read_centroids(path):
    with path.open() as handle:
        return [[float(value) for value in line.rstrip().split(",")]
                for line in handle if line.strip()]


def run_impl(name, data_path, k, max_iter, seed, fixed, centroid_path,
             ranks=1, threads=1, timeout=300):
    command = [str(BINARIES[name]), str(data_path), str(k), str(max_iter), str(seed)]
    if fixed:
        command.append("--fixed-iterations")
    command.extend(["--centroids", str(centroid_path)])

    env = os.environ.copy()
    if name == "openmp":
        env["OMP_NUM_THREADS"] = str(threads)
    elif name == "hybrid":
        env["OMP_NUM_THREADS"] = str(threads)
    if name in ("mpi", "hybrid") and ranks > 1:
        command = [MPI_LAUNCHER, "-np", str(ranks)] + command

    process = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if process.returncode != 0:
        raise TestFailure(
            f"{name} failed with exit code {process.returncode}\n"
            f"command: {' '.join(command)}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    metrics = parse_metrics(process.stdout)
    metrics["centroids"] = read_centroids(centroid_path)
    return metrics


def compare(reference, candidate, label, centroid_atol=1e-7):
    for key in ("rows", "k", "max_iter", "iterations", "converged",
                "stopping_mode", "cluster_sizes", "label_hash"):
        require(
            reference[key] == candidate[key],
            f"{label}: {key} differs: {reference[key]!r} != {candidate[key]!r}",
        )
    require(
        math.isclose(reference["inertia"], candidate["inertia"], rel_tol=1e-9, abs_tol=1e-4),
        f"{label}: inertia differs: {reference['inertia']} != {candidate['inertia']}",
    )
    require(
        len(reference["centroids"]) == len(candidate["centroids"]) == reference["k"],
        f"{label}: centroid row count differs",
    )
    require(
        all(len(row) == 54 for row in reference["centroids"] + candidate["centroids"]),
        f"{label}: centroid feature count differs",
    )
    max_difference = max(
        abs(expected - actual)
        for expected_row, actual_row in zip(reference["centroids"], candidate["centroids"])
        for expected, actual in zip(expected_row, actual_row)
    )
    require(
        max_difference <= centroid_atol,
        f"{label}: maximum centroid difference {max_difference} exceeds {centroid_atol}",
    )
    return max_difference


def write_mixture(path, rows=503, seed=5208):
    rng = random.Random(seed)
    groups = 7
    with path.open("w") as handle:
        for i in range(rows):
            group = i % groups
            values = []
            for feature in range(10):
                offset = 4.0 * group if feature % 2 == 0 else -2.0 * group
                values.append(f"{offset + rng.gauss(0.0, 0.35):.10f}")
            for feature in range(44):
                probability = 0.08 + 0.12 * ((group + feature) % 7)
                values.append("1" if rng.random() < probability else "0")
            values.append(str(group + 1))
            handle.write(",".join(values) + "\n")


def execute_suite(real_data, timeout):
    with tempfile.TemporaryDirectory(prefix="kmeans-tests-") as temp_name:
        temp = Path(temp_name)
        small_data = temp / "mixture.data"
        write_mixture(small_data)

        def case(name, data, k, max_iter, seed, fixed, ranks=1, threads=1):
            return run_impl(
                name, data, k, max_iter, seed, fixed,
                temp / f"{name}-{ranks}x{threads}-{k}-{max_iter}-{fixed}.centroids",
                ranks=ranks, threads=threads, timeout=timeout,
            )

        print("small K=7 fixed-iteration matrix and nondivisible partitions")
        require(small_data.read_text().count("\n") % 3 == 2, "fixture must not divide by 3")
        require(small_data.read_text().count("\n") % 4 == 3, "fixture must not divide by 4")
        baseline = case("serial", small_data, 7, 12, 42, True)
        require(baseline["iterations"] == 12, "fixed mode did not execute MAX_ITER iterations")
        require(baseline["stopping_mode"] == "fixed_iterations",
                "fixed mode reported the wrong stopping mode")
        differences = []
        for threads in (1, 2, 4):
            differences.append(compare(
                baseline, case("openmp", small_data, 7, 12, 42, True, threads=threads),
                f"OpenMP 1x{threads}",
            ))
        for ranks in (1, 2, 3, 4):
            differences.append(compare(
                baseline, case("mpi", small_data, 7, 12, 42, True, ranks=ranks),
                f"MPI {ranks}x1",
            ))
        hybrid_layouts = [(1, 4), (2, 2), (4, 1)]
        if os.cpu_count() and os.cpu_count() >= 8:
            hybrid_layouts.append((2, 4))
        for ranks, threads in hybrid_layouts:
            result = case("hybrid", small_data, 7, 12, 42, True,
                          ranks=ranks, threads=threads)
            require(int(result["ranks"]) == ranks, "reported MPI rank count is wrong")
            require(int(result["threads"]) == threads, "reported OpenMP thread count is wrong")
            require(int(result["total_workers"]) == ranks * threads,
                    "reported total worker count is wrong")
            differences.append(compare(baseline, result, f"Hybrid {ranks}x{threads}"))
        print(f"  matched; max centroid difference {max(differences):.3e}")

        print("small K=1 and default convergence stopping")
        k1 = case("serial", small_data, 1, 20, 17, False)
        require(k1["converged"] and k1["iterations"] < 20, "K=1 did not converge early")
        for name, ranks, threads in (("openmp", 1, 2), ("mpi", 3, 1), ("hybrid", 2, 2)):
            compare(k1, case(name, small_data, 1, 20, 17, False, ranks, threads),
                    f"K=1 {name} {ranks}x{threads}")

        print("more MPI ranks than observations")
        tiny_data = temp / "tiny.data"
        write_mixture(tiny_data, rows=3)
        tiny_k1 = case("serial", tiny_data, 1, 5, 3, True)
        compare(tiny_k1, case("mpi", tiny_data, 1, 5, 3, True, ranks=4), "tiny MPI 4x1")
        compare(tiny_k1, case("hybrid", tiny_data, 1, 5, 3, True, ranks=4, threads=1),
                "tiny hybrid 4x1")

        converged = case("serial", small_data, 7, 100, 7, False)
        require(converged["converged"], "small K=7 did not converge")
        require(converged["stopping_mode"] == "convergence",
                "default stopping mode is not convergence")
        compare(converged, case("mpi", small_data, 7, 100, 7, False, ranks=3),
                "convergence MPI 3x1")
        compare(converged, case("hybrid", small_data, 7, 100, 7, False, ranks=2, threads=2),
                "convergence hybrid 2x2")

        print("real Covertype K=7 fixed iterations")
        require(real_data.is_file(), f"missing real dataset: {real_data}")
        real_fixed = case("serial", real_data, 7, 5, 42, True)
        require(real_fixed["rows"] == 581012, "unexpected Covertype row count")
        differences = []
        differences.append(compare(
            real_fixed,
            case("openmp", real_data, 7, 5, 42, True, threads=2),
            "Covertype fixed OpenMP 1x2",
        ))
        differences.append(compare(
            real_fixed,
            case("mpi", real_data, 7, 5, 42, True, ranks=3),
            "Covertype fixed MPI 3x1",
        ))
        differences.append(compare(
            real_fixed,
            case("hybrid", real_data, 7, 5, 42, True, ranks=2, threads=2),
            "Covertype fixed Hybrid 2x2",
        ))
        print(f"  matched; max centroid difference {max(differences):.3e}")

        print("real Covertype K=1 convergence")
        real_k1 = case("serial", real_data, 1, 10, 42, False)
        for name, ranks, threads in (("openmp", 1, 2), ("mpi", 3, 1), ("hybrid", 2, 2)):
            compare(real_k1, case(name, real_data, 1, 10, 42, False, ranks, threads),
                    f"Covertype K=1 {name} {ranks}x{threads}")

        print("real Covertype K=7 convergence")
        real_converged = case("serial", real_data, 7, 100, 42, False)
        require(real_converged["converged"], "Covertype K=7 did not converge")
        for name, ranks, threads in (("openmp", 1, 2), ("mpi", 3, 1), ("hybrid", 2, 2)):
            compare(real_converged,
                    case(name, real_data, 7, 100, 42, False, ranks, threads),
                    f"Covertype K=7 convergence {name} {ranks}x{threads}")


def run_empty_cluster_test():
    compiler = os.environ.get("MPICXX", "mpicxx")
    binary = Path(tempfile.gettempdir()) / "kmeans_empty_cluster_test"
    process = subprocess.run(
        [compiler, "-O2", "-std=c++17", str(ROOT / "tests/empty_cluster_test.cpp"),
         "-o", str(binary)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    require(process.returncode == 0, f"empty-cluster test compile failed:\n{process.stderr}")
    process = subprocess.run(
        [str(binary)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    require(process.returncode == 0, f"empty-cluster test failed:\n{process.stderr}")
    print(process.stdout.strip())


def execute_serial_openmp_suite(timeout):
    with tempfile.TemporaryDirectory(prefix="kmeans-basic-tests-") as temp_name:
        temp = Path(temp_name)
        small_data = temp / "mixture.data"
        write_mixture(small_data)
        baseline = run_impl(
            "serial", small_data, 7, 12, 42, True,
            temp / "serial.centroids", timeout=timeout,
        )
        for threads in (1, 2, 4):
            candidate = run_impl(
                "openmp", small_data, 7, 12, 42, True,
                temp / f"openmp-{threads}.centroids", threads=threads, timeout=timeout,
            )
            compare(baseline, candidate, f"basic OpenMP 1x{threads}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data", type=Path, default=ROOT / "data/covtype.data")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--skip-mpi", action="store_true",
                        help="run serial/OpenMP checks only (MPI coverage is skipped)")
    args = parser.parse_args()

    global MPI_LAUNCHER
    MPI_LAUNCHER = shutil.which("mpirun") or shutil.which("mpiexec")

    required = ("serial", "openmp") if args.skip_mpi else tuple(BINARIES)
    missing = [name for name in required if not BINARIES[name].is_file()]
    if missing:
        print(f"ERROR: missing binaries: {', '.join(missing)}; run make first", file=sys.stderr)
        return 1

    if args.skip_mpi:
        try:
            execute_serial_openmp_suite(args.timeout)
        except (TestFailure, subprocess.TimeoutExpired) as error:
            print(f"FAIL: {error}", file=sys.stderr)
            return 1
        print("SKIP: MPI and hybrid tests disabled by --skip-mpi")
        print("PASS: serial and OpenMP comparisons succeeded")
        return 0
    if not MPI_LAUNCHER:
        print("ERROR: mpirun/mpiexec not found; install MPI or pass --skip-mpi", file=sys.stderr)
        return 1

    try:
        run_empty_cluster_test()
        execute_suite(args.real_data.resolve(), args.timeout)
    except (TestFailure, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print("PASS: serial, OpenMP, MPI, and hybrid comparisons succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
