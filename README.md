# Covertype K-means: serial, OpenMP, MPI, and hybrid

This repository contains four C++17 implementations of the same Lloyd
K-means algorithm:

- `kmeans_serial.cpp`: serial reference.
- `kmeans_openmp.cpp`: one process with OpenMP threads.
- `kmeans_mpi.cpp`: distributed-memory MPI.
- `kmeans_hybrid.cpp`: MPI ranks with OpenMP teams inside each rank.

Each program reads the UCI Covertype CSV with 54 numerical features and one
known class label. The label is excluded from clustering. The programs use
the same data validation, z-score preprocessing for the first 10 features,
`std::mt19937_64`-seeded K-means++ initialization, squared Euclidean
distance, lowest-cluster-ID tie breaking, empty-cluster policy, and
label-change convergence rule.

## Build

The Linux commands use the normal MPI compiler wrapper:

```sh
g++ -O3 -std=c++17 -DNDEBUG kmeans_serial.cpp -o kmeans_serial
g++ -O3 -std=c++17 -DNDEBUG -fopenmp kmeans_openmp.cpp -o kmeans_openmp
mpicxx -O3 -std=c++17 -DNDEBUG kmeans_mpi.cpp -o kmeans_mpi
mpicxx -O3 -std=c++17 -DNDEBUG -fopenmp kmeans_hybrid.cpp -o kmeans_hybrid
```

The provided `Makefile` builds all four and has a Darwin-specific OpenMP
path for a Homebrew `libomp` installation:

```sh
make
make test
```

Use `CXX`, `MPICXX`, `BASE_FLAGS`, or `OPENMP_FLAGS` to override local
compiler settings. Match compiler optimization and architecture flags when
collecting scaling results.

## Execution

The positional arguments are:

```text
DATA.csv [K=7] [MAX_ITER=100] [SEED=42]
```

Examples:

```sh
./kmeans_serial data/covtype.data 7 100 42
OMP_NUM_THREADS=4 ./kmeans_openmp data/covtype.data 7 100 42
mpirun -np 4 ./kmeans_mpi data/covtype.data 7 100 42
OMP_NUM_THREADS=4 mpirun -np 2 ./kmeans_hybrid data/covtype.data 7 100 42
```

Two optional flags are supported by all four programs:

```text
--fixed-iterations    Execute exactly MAX_ITER iterations; do not stop early.
--centroids FILE      Write the final centroid matrix at full precision.
```

For example:

```sh
mpirun -np 4 ./kmeans_mpi data/covtype.data 7 100 42 \
  --fixed-iterations --centroids mpi-centroids.csv
```

Centroid output files contain K rows and 54 comma-separated coordinates.
Only MPI rank 0 writes this file. The original default behavior is unchanged
when the flags are omitted.

## Common algorithm

Rank 0 or the serial process loads the complete dataset and validates every
row. Columns 0 through 53 are stored in a contiguous row-major
`std::vector<double>`; column 54 is discarded. `scale_continuous()` uses the
existing online mean/M2 calculation, divides by population standard
deviation, and maps a zero-variance feature to zero. The remaining 44 binary
features are unchanged.

Rank 0 performs the existing seeded K-means++ initialization over the full
preprocessed dataset. Because this happens before distribution, every rank
starts from bitwise-identical centroids. During each Lloyd iteration, points
are assigned to the nearest centroid, nonempty clusters are replaced by the
arithmetic mean of their observations, and empty clusters retain their
previous centroid. Convergence means that the globally reduced number of
changed assignments is zero.

## Pure MPI design

### Observation ownership

Rank 0 owns the complete matrix only for loading, preprocessing,
initialization, and the initial distribution. It partitions rows into
contiguous near-equal subsets:

```text
base = rows / ranks
remainder = rows % ranks
local_rows(rank) = base + (rank < remainder ? 1 : 0)
```

`MPI_Scatterv` sends each row block, including its `local_rows * 54` double
elements. This supports any positive number of ranks and row counts that are
not divisible by the rank count. Rank 0 then releases its complete matrix;
non-root ranks never hold all observations.

### Collective operations

Every rank executes the same collective sequence:

1. Broadcast the row count.
2. `MPI_Scatterv` the preprocessed observation rows.
3. `MPI_Bcast` the initial centroid array.
4. In each iteration, `MPI_Allreduce` cluster feature sums with `MPI_SUM`.
5. In each iteration, `MPI_Allreduce` cluster counts and the local changed
   label count with `MPI_SUM`.
6. After fitting, reduce local inertia once with `MPI_SUM`.
7. `MPI_Gatherv` final labels in rank order so rank 0 can reproduce the serial
   label hash without communicating labels every iteration.

Each rank computes the same updated centroids from the global sums and
counts. Centroid updates therefore require an all-rank barrier-like
collective every iteration. Labels remain local until final validation.

## Hybrid MPI + OpenMP design

The hybrid uses `MPI_Init_thread` to request `MPI_THREAD_FUNNELED` and rejects
a runtime that provides less support. All MPI calls are made by the main
thread, outside OpenMP parallel regions.

The data distribution is identical to the pure MPI implementation. Within
each rank, `OMP_NUM_THREADS` controls an OpenMP team:

1. Allocate a centroid-sum block and a count block per thread.
2. Use `#pragma omp parallel` and `#pragma omp for schedule(static)` over the
   rank's local observations.
3. Each thread updates only its own labels and private totals. Centroids are
   read-only during assignment, so the loop has no atomics or critical
   sections.
4. After the implicit parallel-region barrier, the main thread combines
   thread totals in fixed thread-ID order.
5. `MPI_Allreduce` combines those rank-local totals into global totals.
6. Every rank performs the same centroid update from the global result.

This is the two-level reduction. Level 1 is a deterministic sequential
reduction of OpenMP thread accumulators. Level 2 is MPI all-reduction across
ranks. The reported `threads` value is the actual OpenMP team size after
dynamic team resizing is disabled; `total_workers` is summed across ranks.
Configurations such as `1x4`, `2x2`, `4x1`, `2x4`, and `4x4` are supported.

## Why the parallel versions are not embarrassingly parallel

Distance calculations and local accumulation are independent across
observations, which is the parallel part. A centroid, however, must be
updated from all observations in its cluster, including observations owned
by other MPI ranks. The cluster sums, counts, and convergence decision must
therefore be combined globally after every iteration, and every rank must
finish reading the old centroids before the next iteration starts. Those
global reductions and synchronization are the distributed-memory bottleneck;
additional OpenMP barriers add a smaller local overhead.

## Output and timing definitions

Only rank 0 prints the final human-readable values, cluster sizes, label hash,
CSV header, and CSV record. The common fields are:

- `rows`, `features`, `k`, `max_iter`, `seed`
- `iterations`, `inertia`
- `load_s`, `preprocess_s`, `init_s`, `fit_s`, `validation_s`, `total_s`
- `stopping_mode`, `converged`
- `cluster_sizes`, `label_hash`

MPI programs additionally print:

- `implementation`: `mpi` or `hybrid`
- `ranks`, `threads`, `total_workers`
- `distribution_s`
- `mpi_reduction_s`
- `local_compute_s`

Timing scopes are:

- `load_s`, `preprocess_s`, and `init_s` are rank-0-only serial phases.
- `distribution_s` is wall time from the start of distribution through the
  row scatter and centroid broadcast, reported as the maximum across ranks.
- `fit_s` is the maximum wall-clock duration of the complete Lloyd loop,
  including local assignment, both reductions, synchronization, and centroid
  updates. It is the primary performance metric.
- `local_compute_s` is cumulative rank-local assignment/accumulation time,
  including the hybrid's OpenMP thread reduction, reported as the maximum.
- `mpi_reduction_s` is cumulative time spent inside the per-iteration MPI
  all-reductions, reported as the maximum. It excludes collective timing
  reductions and final validation.
- `validation_s` covers final local inertia, inertia reduction, and label
  gathering, reported as the maximum.
- `total_s` is the maximum rank wall time from entry to fitting into `run()`
  through validation. It excludes final text/CSV printing.

These separately measured components are not assumed to sum exactly to
`fit_s`; barriers, waits, resets, centroid arithmetic, and timing calls are
also present.

The CSV header is:

```text
implementation,ranks,threads,total_workers,rows,features,k,max_iter,iterations,fit_s,sse,converged,stopping_mode,distribution_s,mpi_reduction_s,local_compute_s
```

The final assignments are represented by `label_hash`, which follows the
original row order and is recomputed after gathering labels. Cluster sizes
also describe the complete dataset, not a single rank.

## Correctness tests

Build all binaries, then run:

```sh
python3 tests/run_tests.py
```

The test script compares final inertia, all centroid coordinates, cluster
sizes, iteration count, convergence status, stopping mode, and the complete
assignment sequence through `label_hash`. It uses numerical tolerances for
floating-point values and exact checks for counts and labels. The suite
covers:

- A deterministic 503-row mixture dataset.
- Real `data/covtype.data`.
- K=1 and K=7.
- Row counts not divisible by 3 or 4 MPI ranks.
- OpenMP thread counts 1, 2, and 4.
- MPI rank counts 1, 2, 3, and 4.
- Hybrid configurations `1x4`, `2x2`, and `4x1`, plus `2x4` when at least
  eight CPU threads are available.
- More MPI ranks than observations, with zero-row ranks.
- Default convergence and fixed `MAX_ITER` stopping.
- A direct unit test of the empty-cluster centroid path.

`python3 tests/run_tests.py --skip-mpi` runs the serial/OpenMP subset when MPI
is unavailable. The full suite fails clearly rather than silently skipping
missing MPI or hybrid binaries.

### Executed validation results

The full suite was executed locally on 2026-11-04 with Apple Clang 21,
MPICH 5.0.2, and Homebrew libomp 18.1.3. All checks passed:

- The small K=7 fixed-iteration matrix matched across serial, OpenMP,
  MPI, and all tested hybrid layouts. Maximum centroid difference was
  `1.110e-15`.
- Real Covertype with 581,012 rows, K=7, seed 42, and five fixed iterations
  matched across serial, OpenMP 1x2, MPI 3x1, and hybrid 2x2. Maximum
  centroid difference was `2.706e-13`.
- Real Covertype K=1 and full K=7 convergence runs matched in iteration
  count, inertia, cluster sizes, convergence status, and label hash.
- The K=7 full-convergence reference reached 91 iterations with inertia
  `3712497.960091` and label hash `9039984004050467668`.
- The direct empty-cluster test confirmed that a zero-count cluster keeps
  its previous centroid while a nonempty cluster receives its arithmetic
  mean.

These are correctness checks, not final Atlas/Vanda scaling measurements.

## Known limitations

- Loading, preprocessing, and K-means++ remain serial on rank 0. This keeps
  initialization exactly comparable and is important because the dataset is
  loaded once rather than ingested in parallel.
- Every iteration requires two collective reductions. A production system
  might combine the sum, count, and changed-label payloads or use persistent
  collectives, but those optimizations are intentionally omitted for
  readability.
- Floating-point reduction order can change low-order bits when the number
  of ranks or threads changes. This can eventually alter a tie or convergence
  iteration on a near-boundary dataset. Fixed-iteration mode is provided for
  stricter performance comparisons.
- Final labels are gathered once for hash validation. Labels are not
  replicated during the fitting loop.
- Performance depends on the MPI implementation, process binding, thread
  affinity, memory bandwidth, and node topology. Run repeated trials with
  matched builds and avoid oversubscribing ranks times threads beyond the
  available CPU resources.
