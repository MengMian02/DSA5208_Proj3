# Covertype K-means: serial, OpenMP, MPI, hybrid and Apple Metal

Parallel scaling study (DSA5208 Project 3) of Lloyd's K-means on the UCI
Covertype dataset: 54 features, 581012 observations, classification label
excluded, K = 7 by default.

Five implementations share one algorithm definition through
kmeans_common.hpp:

| Program | Parallelism | Notes |
|---|---|---|
| kmeans_serial | none | reference implementation |
| kmeans_openmp | OpenMP threads | thread-local accumulation, schedule(static) |
| kmeans_mpi | MPI ranks | observation-wise decomposition, MPI_Allreduce, optional deterministic weak-scaling generation |
| kmeans_hybrid | MPI ranks x OpenMP threads | MPI_Init_thread(MPI_THREAD_FUNNELED) |
| kmeans_metal | Apple GPU (Metal) | local-only, macOS + Apple Silicon |
| kmeans_cuda | NVIDIA GPU (CUDA) | gpu/cuda/, validated against the CPU reference on one A40 |

## Algorithm and shared conventions

Identical in every CPU implementation, preserved from the original code:

* C++17, constexpr int D = 54, contiguous row-major std::vector<double>.
* 55-column Covertype CSV; column 55 is the class label and is discarded.
* z-score of the first 10 continuous features (population standard deviation,
  zero-variance feature maps to 0); the 44 binary features are untouched.
* K-means++ initialization seeded by std::mt19937_64.
* Squared Euclidean distance, lowest-cluster-id tie breaking.
* Lloyd assignment and centroid update; **empty clusters keep their previous
  centroid**.
* Convergence = zero changed assignment labels across the whole dataset.
* Output metrics: rows, features, k, max_iter, seed, iterations, inertia,
  load_s, preprocess_s, init_s, fit_s, validation_s, total_s, converged,
  cluster_sizes, label_hash, plus csv_header=/csv= machine-readable lines.

### Timing boundaries

fit_s is the primary metric: it covers the Lloyd iterations **only**. Loading
(load_s), preprocessing (preprocess_s), k-means++ (init_s) and the final
inertia/label validation (validation_s) are measured separately and never
folded into fit_s.

For MPI, fit_s is the **maximum** elapsed fitting time across ranks
(MPI_Wtime measured around the iteration loop), never a sum of rank times. The
minimum across ranks and their difference (rank imbalance) are reported too.

### Phase instrumentation (E5)

Every implementation reports the same phase keys:

| Key | Meaning |
|---|---|
| local_compute_s | nearest-centroid assignment + local accumulation (max across ranks) |
| mpi_reduction_s | MPI_Allreduce time: collective latency **including** waiting for slower ranks, not pure network transfer |
| thread_reduction_s | folding thread-local totals into rank-local totals |
| centroid_update_s | mean computation over K x D |
| distribution_s | row scatter/broadcast + initial centroid broadcast + (weak mode) local generation |
| fit_min_s, fit_imbalance_s | rank spread, the load-imbalance indicator |

## Build

    make            # all CPU programs plus kmeans_metal on macOS
    make cpu        # serial, OpenMP, MPI, hybrid
    make gpu        # Apple Metal (macOS only)
    make test       # CPU cross-checks, weak-scaling checks, Metal checks
    make smoke      # 30-second end-to-end check

Local toolchain used for the measurements in docs/: Apple clang 21, OpenMPI
5.0.2 (Homebrew), -O3 -std=c++17 -DNDEBUG.

On macOS the Makefile picks up Homebrew libomp automatically. The Metal program
needs only the Command Line Tools: the shader is compiled from an embedded
source string at run time (see gpu/metal/embed_source.py), so xcrun metal and
full Xcode are not required. Override CXX, MPICXX, BASE_FLAGS, OPENMP_FLAGS or
OPENMP_LIBS as needed, and record whatever you use in BENCH_FLAGS so the CSV
rows carry the real flags.

## Run

    DATA.csv [K=7] [MAX_ITER=100] [SEED=42]
        [--fixed-iterations]        run exactly MAX_ITER iterations (benchmarking)
        [--centroids FILE]          write the final centroid matrix at full precision
        [--rows N]                  systematic subsample of N observations
        [--init-centroids FILE]     use fixed initial centroids instead of k-means++
        [--weak-per-rank R]         MPI only: deterministic weak-scaling workload
        [--source-bin FILE]         MPI weak mode: cached binary source instead of a broadcast
        [--dump-generated FILE]     MPI weak mode: write the generated rows (verification)

kmeans_metal adds --mode assign|full, --bench-overhead and --validate.

    ./kmeans_serial data/covtype.data 7 30 42 --fixed-iterations --rows 100000
    OMP_NUM_THREADS=4 ./kmeans_openmp data/covtype.data 7 30 42 --fixed-iterations
    mpirun -np 4 ./kmeans_mpi data/covtype.data 7 30 42 --fixed-iterations
    OMP_NUM_THREADS=2 mpirun -np 4 ./kmeans_hybrid data/covtype.data 7 30 42 --fixed-iterations
    ./kmeans_metal data/covtype.data 7 30 42 --rows 100000 --fixed-iterations --mode full

--rows N keeps N evenly spaced rows of the loaded file, so every implementation
and every rank count sees the same subset in the same order. Preprocessing is
then recomputed on that subset, which is the correct definition of K-means on a
smaller dataset.

## Deterministic weak-scaling generation (E3)

Weak scaling needs a larger logical dataset per rank without writing expanded
files. kmeans_mpi --weak-per-rank R builds rank r's contiguous block of global
indices [r*R, (r+1)*R) in memory from the preprocessed Covertype source:

    source_index(seed, gi) =
        splitmix64( gi * 0x9E3779B97F4A7C15 ^ splitmix64(seed ^ 0x5DEECE66D) ) mod source_rows

splitmix64 is the standard fixed integer mixer (no std::hash, no
library-dependent behaviour), implemented once in kmeans_common.hpp and
re-implemented independently in tests/run_weak_scaling_tests.py, which verifies
the materialised rows to 1e-12. The mapping depends only on
(seed, global index, source rows), never on the rank count, so 1 rank x 6000 and
4 ranks x 1500 produce the identical logical dataset and the identical
clustering. The resulting work is an **expanded benchmark workload built by
resampling real Covertype rows**, not new observations.

Initial centroids are k-means++ on the *source* Covertype matrix, so every rank
count starts from the same centroids. The source matrix reaches the ranks either
by one chunked MPI_Bcast (default) or by a cached binary file (--source-bin),
which avoids a 252 MB broadcast at high rank counts. Both paths are measured and
reported separately (source_bcast_s, load_s, generation_s).

## Experiments and tooling

| Script | Experiment |
|---|---|
| scripts/run_e1.py | E1 serial scaling over N and K |
| scripts/run_mpi.py --experiment strong | E2 strong scaling |
| scripts/run_mpi.py --experiment weak | E3 weak scaling |
| scripts/run_mpi.py --experiment hybrid --layouts 72x1,36x2,... | E4 rank/thread layouts |
| scripts/run_metal.py | E6 Metal |
| scripts/run_cuda.py | E7 CUDA assignment kernel (Vanda A40) |
| scripts/analyze.py | all figures + results/SUMMARY.md |
| scripts/smoke_test.py | build + cross-implementation agreement |

Each runner probes the environment (host, CPU model, cores, compiler, MPI
version, flags), executes the configuration, parses the machine-readable output,
and **appends** one row per repetition to results/*.csv. Failed or incomplete
runs are refused and logged to results/failures.log; an existing CSV is never
truncated, and a schema mismatch aborts instead of mixing data.

    python3 scripts/run_e1.py --repetitions 3
    python3 scripts/run_mpi.py --experiment strong --ranks 1,2,4,8 --repetitions 3
    python3 scripts/run_mpi.py --experiment weak   --ranks 1,2,4,8 --per-rank 50000 --repetitions 3
    python3 scripts/run_mpi.py --experiment hybrid --layouts 4x1,4x2,2x2,2x4,1x4 --repetitions 3
    python3 scripts/run_metal.py --rows 20000,100000,400000 --repetitions 3
    python3 scripts/analyze.py

scripts/analyze.py writes figures/*.png and results/SUMMARY.md. An experiment
with no measured data produces a figure that says **PENDING**; no value is ever
invented or interpolated.

Atlas runs are analysed the same way, from the local machine:

    python3 scripts/analyze.py --results results/atlas --figures figures/atlas \
        --experiments e1,e2,e3,e4,e5

The Atlas login node ships no pandas or matplotlib (system python3 is 3.6.8 and
the python3.7 module has no site-packages for them), so plotting stays where
the libraries exist. The Python tools are 3.6 compatible and print that
instruction instead of failing with an import or syntax error; the C++ programs
and the PBS job scripts are unaffected.

## Tests

    make test

* tests/run_tests.py - serial/OpenMP/MPI/hybrid equality on a synthetic mixture
  and on real Covertype: K = 1 and K = 7, fixed-iteration and convergence
  stopping, partitions that do not divide the row count, more ranks than
  observations, several thread counts, empty-cluster policy.
* tests/run_weak_scaling_tests.py - the documented index mapping reproduced by
  an independent implementation, rank-count independence of the logical
  dataset, repeat determinism, cached-source equivalence.
* tests/run_metal_tests.py - Stage A/B agreement, GPU-vs-CPU label agreement and
  inertia tolerance, K = 1, convergence decision, determinism. Skips cleanly when
  Metal is unavailable.

Float reductions across MPI ranks or GPU threads are **not** bitwise identical
to serial summation; the tests therefore compare centroids, inertia, cluster
sizes and label hashes with tolerances, and treat a differing clustering as a
failure until it is explained.

## HPC

jobs/ holds the Atlas and Vanda submission kits. Both clusters run **PBS Pro**,
not SLURM. docs/HPC_LOG.md records every job and its outcome; the measured CSVs
are in results/atlas/, results/vanda/ and results/vanda_ours/.

Vanda specifics that cost us several failed submissions and are worth knowing:

* Execution queues are `from_route_only`. Route `auto` sends a job to
  `batch_cpu` or `batch_gpu`; route `auto_free` sends it to `cpu_serial`,
  `cpu_parallel` or `gpu`. Submitting to an execution queue directly is refused.
* `select=N` means N **chunks**, not N hosts, so every multi-node job sets
  `place=scatter`. Multi-node Open MPI needs `/opt/pbs/lib` on
  `LD_LIBRARY_PATH` to use the PBS TM launch, and must not be given
  `--hostfile`.
* PBS derives memory from core count (~6.96 GB per core), so a 64-core chunk
  demands a nearly empty 72-core node. That, not oversubscription, is why
  256-core jobs wait for a long time.
* Open MPI 4.1.8 cannot reserve cores per rank here: every `pe=` mapping form
  (`ppr:N:node:pe=T`, `slot:pe=T`, `node:pe=T`) hangs under the TM launch and
  `core:pe=T` is rejected. Only one-thread-per-rank layouts can be bound with
  `--map-by core --bind-to core`; multi-thread layouts run unpinned and their
  collectives are then dominated by scheduler-induced waiting. This is the
  single most important caveat when reading the hybrid (E4) numbers.

Measured on Vanda, job IDs in docs/HPC_LOG.md: E1 serial (1546646), E2 strong
1->256 ranks on two different node sets (1551056 on GN-A40, 1563107 on CN),
E3 weak 1->256 (1551057, 1551070), E4 single node at 64 cores (1551076) and E4
at 256 workers pinned and unpinned (1556850, 1554129), E7 CUDA on one A40
(1554475). Atlas was capped at 96 ranks per user, so the 256-core requirement is
met from Vanda.

## Repository layout

    kmeans_common.hpp        shared algorithm, options, timing and generator code
    kmeans_*.cpp             the four CPU programs
    gpu/metal/               Metal shader, embedded-source generator, Objective-C++ driver
    gpu/cuda/                CUDA assignment kernel for the Vanda A40 experiment
    scripts/                 benchmark runners, analysis, smoke test
    jobs/                    Atlas/Vanda submission kit
    tests/                   correctness, weak-scaling, Metal and MPI Allreduce probes
    results/                 measured CSV records + SUMMARY.md (vanda/, vanda_ours/, atlas/)
    figures/                 generated plots (vanda/, atlas/)
    docs/                    experiment manifest, implementation report, HPC log
    supplementary/           team's supplementary HPC package (own build, jobs, results)
    data/                    Covertype data set + documentation

## Supplementary HPC Experiments

The team conducted additional experiments to investigate large-workload MPI
scaling, OpenMP thread scaling, Hybrid MPI+OpenMP rank/thread configurations,
and MPI process placement and communication overhead. The reproducible package,
including its own build, scripts, PBS jobs, verified CSVs and validation
evidence, is documented in [`supplementary/README.md`](supplementary/README.md).
