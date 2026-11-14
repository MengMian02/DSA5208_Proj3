# Implementation and validation report

Date: 2026-11-05. Author: coding agent. Scope: E1-E6 framework for the DSA5208
K-means scaling study. This document states exactly what is finished, what is
merely prepared, and what is missing. No unexecuted benchmark is reported as a
result.

## 1. Starting point found during the audit

The repository already contained working serial, OpenMP, MPI and hybrid
implementations (kmeans_serial.cpp, kmeans_openmp.cpp, kmeans_mpi.cpp,
kmeans_hybrid.cpp), a Makefile, a README, tests/run_tests.py and an
empty-cluster unit test. They were not missing and were not rewritten from
scratch: the audit found them consistent (identical preprocessing, k-means++,
empty-cluster policy and convergence rule) and the original git history shows
serial+OpenMP, then MPI+hybrid.

What was missing: a shared core, systematic-subset support, deterministic weak
scaling, uniform E5 phase instrumentation, any benchmark automation, any HPC
submission kit, any plotting, and the whole Apple Metal implementation.

Local environment measured during the audit: Apple M1 Pro, 8 cores (6
performance + 2 efficiency), 16 GB unified memory, Apple clang 21, OpenMPI
5.0.2, Homebrew libomp, Metal available (device Apple M1 Pro, unified memory),
but no xcrun metal because only the Command Line Tools are installed.

## 2. Status of the deliverables

### Completed and tested

| Item | Evidence |
|---|---|
| Shared core kmeans_common.hpp (options, subset reader, z-score, k-means++, distance, centroid update, inertia, label hash, splitmix64 generator, cached binary source) | all five programs build from it; cpu tests pass |
| kmeans_serial, kmeans_openmp refactored onto the shared core with --rows and unified metrics | label_hash matches the pre-refactor binaries and each other |
| kmeans_mpi: strong mode, weak mode, E5 phase timers, --dump-generated | tests/run_weak_scaling_tests.py passes; strong mode still bitwise equal to serial |
| kmeans_hybrid: same plus thread_reduction_s and thread-count verification | 1x1,1x2,2x1,2x2,4x1,2x4,4x2 layouts match serial bitwise |
| Apple Metal Stage A (assignment kernel, naive transfers) and Stage B (GPU-resident matrix, per-iteration centroid upload) | tests/run_metal_tests.py passes: label agreement 1.0, centroid diff 0, K=1, convergence decision, determinism |
| scripts/bencher.py + run_e1.py + run_mpi.py + run_metal.py (metadata, parsing, append-only CSV, failure log) | 113 rows recorded across 5 CSVs, 0 failures |
| scripts/analyze.py (E1-E6 figures + results/SUMMARY.md, PENDING placeholders) | 10 figures generated from measured data |
| scripts/smoke_test.py, tests/run_weak_scaling_tests.py, tests/run_metal_tests.py | make test passes end to end |
| E1, E2 (1-8 ranks), E3 (1-8 ranks), E4 (8-core layouts), E5 breakdown, E6 | measured on this machine; see section 5 |

### Implemented but untested

* The 256-core paths: rank counts 16, 32, 64, 128, 256 have never been executed
  anywhere. The code paths they use (chunked broadcast, Gatherv, ragged blocks)
  are exercised only up to 8 ranks locally.
* --source-bin at high rank counts: verified equal to the broadcast path at 2
  ranks only. Behaviour with 256 ranks reading one 252 MB file from a parallel
  filesystem is unmeasured.
* The 72-core and multi-node hybrid layouts (72x1 ... 1x72): the layout
  machinery is tested, the specific Linux/Atlas executions are not.
* jobs/*.pbs and jobs/*.slurm: bash -n syntax checked only. They have never been
  submitted, so scheduler acceptance of the directives is unverified.

### Prepared for HPC execution (nothing submitted)

* jobs/cluster.env.example + jobs/probe_cluster.sh: the probe collects CPU
  topology, scheduler queues/partitions, module tree, launcher names, compilers,
  filesystem and interconnects. Every site-specific value must come from it; the
  job bodies refuse to run while a value still says CHANGE_ME.
* jobs/common.sh: preflight, module loading, compute-node build, core-binding
  flags per launcher.
* jobs/atlas/{serial,strong_scaling,weak_scaling,hybrid_layouts}.pbs with
  jobs/atlas/submit.sh building the PBS select= request on the command line
  (PBS directives are not shell-expanded).
* jobs/vanda/{serial,strong_scaling,weak_scaling,hybrid_layouts}.slurm with
  jobs/vanda/submit.sh, including the Vanda requirement to build on the compute
  node.
* jobs/README.md: submission plan, resource math for 256 cores, affinity rules,
  status/retrieval commands, and the mandatory 2-rank checkpoint before
  production submission.
* docs/HPC_LOG.md: empty log template; docs/EXPERIMENTS.md: the manifest.

### Blocked by unavailable hardware or access

* Atlas / Vanda allocations: no job was submitted, so no cluster runtime exists.
  The brief requires runtime figures from Atlas or Vanda in the final report;
  that requirement is currently unmet and cannot be met without an allocation.
* Clean strong/weak scaling beyond 8 ranks: this machine has 6 performance + 2
  efficiency cores, so 8 ranks already oversubscribe the fast cores. The
  16-256 rank points are physically out of reach locally.
* Metal command line toolchain (xcrun metal) is absent. Worked around: shaders
  are compiled at run time from an embedded source string, so no full Xcode is
  needed. Precompiled .metallib output is therefore not produced, which is only
  a packaging difference.

### Not yet implemented (all P2 / out of scope)

* GPU-side centroid accumulation (the identified next bottleneck, see 5.6).
* GPU-resident reduction without the CPU accumulate pass.
* CUDA/NVIDIA path (explicitly excluded by the brief).
* OpenMP schedule comparisons, alternative memory layouts, alternative
  clustering algorithms.

## 3. Design decisions

**Shared core, single algorithm definition.** One header holds preprocessing,
k-means++, distance, centroid update with the empty-cluster rule, inertia, the
label hash and the weak-scaling generator. Every program uses it, so the
implementation convention cannot drift between them. This replaced four
near-identical copies, which is a duplication removal, not an architectural
rewrite: no algorithm behaviour changed, and the original outputs are preserved
(same keys, same order, same semantics).

**Timing boundaries are fixed in the code, not in the analysis.** fit_s wraps
the iteration loop only. Loading, preprocessing, initialization and the final
validation are separate timers in every implementation. For MPI, fit_s is the
maximum across ranks; the minimum is reduced separately so fit_imbalance_s is a
measured rank spread rather than a guess.

**Fixed-iteration mode is the benchmarking default.** --fixed-iterations runs
exactly MAX_ITER iterations; convergence is still recorded. All E1-E6 runs use
it, so every configuration performs the same arithmetic and cross-configuration
timing is comparable.

**Systematic subsampling for --rows.** Keeping the first N rows would have
sampled only a few Covertype classes, because the file is class-ordered.
Instead N evenly spaced rows are kept; selection is a pure function of
(source rows, N), so serial, OpenMP, MPI and Metal all see the same subset in
the same order. Preprocessing is recomputed on the subset, which is the correct
definition of K-means on a smaller dataset; the report must therefore not claim
the E1 series is "the same normalization at smaller N".

**Weak scaling without expanded files.** The global-index hash in the README is
the whole contract: rank r materialises global indices [r*R, (r+1)*R), and the
mapping depends only on (seed, index, source rows). Consequences that were
verified rather than assumed: 1x6000, 2x3000 and 4x1500 give the identical
clustering; the materialised rows match an independent Python implementation to
1e-12; the cached-binary and broadcast source paths agree exactly. Initial
centroids are always k-means++ on the source, never re-drawn per configuration.

**Hybrid threading keeps determinism.** Thread-local sum/count arrays are folded
into rank-local totals in fixed thread-ID order, so a fixed thread count gives a
bitwise-reproducible reduction; MPI_THREAD_FUNNELED is requested and verified,
and only the main thread calls MPI. Thread counts are checked to be equal across
ranks before timing starts - otherwise the "total_workers" label would lie.

**Metal staging is honest about what runs on the GPU.** The kernel does nearest
centroid assignment in float32 (Apple GPUs have no IEEE double); the CPU keeps
the observation matrix in double and performs accumulation and the centroid
update. Stage A deliberately re-uploads everything each iteration to expose the
cost of avoidable traffic; Stage B keeps the matrix resident and re-uploads only
K x D centroids. The GPU kernel time is compared against a CPU assignment pass
of the same work in the same run; end-to-end fitting times are compared
separately, and no kernel/whole-fit ratio is presented as speedup.

**Measurement harness.** One row per repetition, appended, never overwritten; a
schema mismatch aborts instead of mixing data; a missing metric or non-zero exit
is refused and logged to results/failures.log. Every row carries host, CPU
model, core count, compiler string, flags, MPI version, dataset, K, D, seed,
iterations, stopping mode, ranks, threads and repetition number.

## 4. Validation evidence (all executed locally)

make test, full transcript excerpts:

    tests/run_tests.py
      empty cluster retained previous centroid; nonempty cluster updated
      small K=7 fixed-iteration matrix and nondivisible partitions
        matched; max centroid difference 1.110e-15
      small K=1 and default convergence stopping
      more MPI ranks than observations
      real Covertype K=7 fixed iterations
        matched; max centroid difference 2.706e-13
      real Covertype K=1 convergence
      real Covertype K=7 convergence
    PASS: serial, OpenMP, MPI, and hybrid comparisons succeeded

    tests/run_weak_scaling_tests.py
      1x200: 200 rows match the documented mapping (max diff 4.4e-16)
      3x200: 600 rows match the documented mapping (max diff 8.9e-16)
      ranks=1 rows=6000 obs/rank=6000 hash=6870209709768084213 inertia=38806.638833
      ranks=2 rows=6000 obs/rank=3000 hash=6870209709768084213 inertia=38806.638833
      ranks=4 rows=6000 obs/rank=1500 hash=6870209709768084213 inertia=38806.638833
      repeated runs bitwise identical; cached source equals broadcast
      strong mode still matches serial bitwise
    PASS: weak-scaling generation, determinism and reporting verified

    tests/run_metal_tests.py
      Stage A and Stage B agree (identical label_hash)
      N=5000 agreement=1.000000 inertia rel.diff=0.000e+00
      N=100000 agreement=1.000000 inertia rel.diff=0.000e+00
      K=1 cluster_sizes=[2000], label_hash matches serial
      convergence: iterations gpu=57 cpu=57, converged=true
      repeated GPU runs bitwise identical
    PASS: Metal assignment and GPU-assisted iteration validated

Coverage against the required correctness list: small deterministic dataset
(synthetic mixture, 503 rows), real Covertype full set, K=1, K=7, uneven
partitions (row counts that do not divide by 3 or 4), more ranks than
observations, thread counts 1/2/4, rank counts 1/2/3/4, fixed-iteration and
convergence stopping, empty-cluster policy - all in tests/run_tests.py; the
weak-scaling generator and the Metal kernel in the two new suites.

Why bitwise equality is not the acceptance criterion: an MPI_SUM over P ranks
and an OpenMP fold over T threads sum in a different order than the serial loop,
so centroids can differ in the last bits. The tests compare centroids
(atol 1e-7), inertia, cluster sizes, iteration count, converged flag and the
label hash, and treat any clustering difference as a failure to be explained
rather than tolerated.

## 5. Measured results on this machine

Apple M1 Pro, 8 cores, -O3 -std=c++17 -DNDEBUG, OpenMPI 5.0.2, medians of 3
repetitions, seed 42, K=7, I=30, fixed-iteration mode. Raw rows:
results/e1_serial.csv, e2_strong.csv, e3_weak.csv, e4_hybrid.csv,
e6_metal.csv. A first, contaminated pass (other work running concurrently) is
kept separately under results/pilot/ and is not used in any figure.

### 5.1 E1 serial scaling

| N | fit_s | load_s | preprocess_s | init_s |
|---|---|---|---|---|
| 50000 | 0.371 | 0.648 | 0.0032 | 0.0085 |
| 100000 | 0.754 | 0.637 | 0.0098 | 0.0205 |
| 200000 | 1.503 | 0.622 | 0.0216 | 0.0339 |
| 400000 | 3.027 | 0.691 | 0.0574 | 0.0691 |
| 581012 | 4.492 | 0.637 | 0.1011 | 0.1023 |

| K | 2 | 4 | 7 | 10 | 16 |
|---|---|---|---|---|---|
| fit_s | 1.689 | 2.983 | 4.492 | 5.900 | 8.603 |

Fitted log-log exponent: 1.01 for N, 0.777 for K. N is linear as expected. K is
sub-linear for a structural reason: each iteration costs K distances of D
multiply-adds per observation plus one D-add accumulation into the winning
cluster, and that accumulation does not grow with K. A linear fit T(K) = a + b K
gives b = 0.52 s per cluster and a = 0.70 s of K-independent work over the 30
iterations, which reproduces the K=7 measurement (4.3 s predicted vs 4.49 s
measured). The K trend is therefore explained by the algorithm, not by noise.

Note that load_s is roughly constant: the whole 75 MB CSV is always parsed, and
--rows compacts afterwards, so dataset construction does not scale with N. It is
correctly excluded from fit_s.

### 5.2 E2 MPI strong scaling (full Covertype)

| ranks | fit_s | speedup | efficiency | local_compute_s | collective_s | collective share |
|---|---|---|---|---|---|---|
| 1 | 4.238 | 1.00 | 100% | 4.238 | 0.0001 | 0.0% |
| 2 | 2.209 | 1.92 | 96% | 2.202 | 0.0242 | 1.1% |
| 4 | 1.093 | 3.88 | 97% | 1.091 | 0.0049 | 0.4% |
| 8 | 0.993 | 4.27 | 53% | 0.778 | 0.3490 | 35.1% |

Distribution (252 MB scatter) measured 0.82-0.92 s and is excluded from fit_s.
The 1-4 rank points are essentially ideal: the reduction payload is only
K*D doubles = 3.0 KB per rank plus (K+1) uint64 per iteration, so there is
almost nothing to communicate. The 8-rank point is not a clean strong-scaling
measurement: this machine has 6 performance + 2 efficiency cores, so 8 ranks put
2 of them on efficiency cores, which both lowers local compute (4.24 -> 0.78 s
is 5.4x, not 8x) and makes every collective wait for the slow ranks (0.35 s,
35% of fitting time). The honest reading is "strong scaling is close to ideal up
to 4 ranks; a 256-core cluster is needed for the rest of the curve".

### 5.3 E3 MPI weak scaling (50000 generated observations per rank)

| ranks | rows | fit_s | weak efficiency | local_compute_s | collective_s | distribution_s |
|---|---|---|---|---|---|---|
| 1 | 50000 | 0.372 | 100% | 0.372 | 0.0001 | 0.796 |
| 2 | 100000 | 0.381 | 97.7% | 0.378 | 0.0058 | 0.857 |
| 4 | 200000 | 0.393 | 94.7% | 0.385 | 0.0094 | 0.885 |
| 8 | 400000 | 0.822 | 45.3% | 0.531 | 0.318 | 1.529 |

E_weak(P) = T(1,N0)/T(P,P*N0). Efficiency is preserved to 4 ranks and collapses
at 8 for exactly the reason identified in 5.1: the 8th rank set oversubscribes
the efficiency cores, so per-rank compute time rises (0.372 -> 0.531 s) and the
collective wait grows. Source distribution is reported separately: 0.80-1.53 s,
of which the broadcast of the 252 MB source dominates, which is why the HPC jobs
default to the cached binary path. No statistical claim about Covertype may be
drawn from E3: those datasets are resampled expansions of the real rows.

### 5.4 E4 hybrid MPI+OpenMP at 8 cores

| layout | fit_s | local_compute_s | collective_s | OpenMP fold_s |
|---|---|---|---|---|
| 1x1 | 4.271 | 4.271 | 0.0001 | 1.3e-05 |
| 1x2 | 2.258 | 2.258 | 0.0001 | 1.2e-05 |
| 1x4 | 1.185 | 1.185 | 0.0001 | 2.9e-05 |
| 2x1 | 2.180 | 2.178 | 0.0213 | 1.2e-05 |
| 2x2 | 1.209 | 1.182 | 0.0325 | 1.9e-05 |
| 2x4 | 0.910 | 0.878 | 0.0600 | 2.4e-05 |
| 4x1 | 1.114 | 1.108 | 0.0119 | 1.3e-05 |
| 4x2 | 0.959 | 0.851 | 0.1607 | 1.9e-05 |

In-node threading is close to linear (1x1 -> 1x4 is 3.60x on 4 cores), and the
OpenMP fold costs at most 3e-5 s, i.e. it is irrelevant. At the same 8-worker
budget the ranking is 2x4 (0.910) < 4x2 (0.959) < 8x1 (0.993 from E2): fewer
ranks with more threads win, and the gain is explained by collective time, not
by the fold. But the collective series is not monotone in rank count (2x1 0.021,
4x1 0.012, 2x4 0.060, 4x2 0.161), which is direct local evidence for the
interpretation rule in the manifest: MPI_Allreduce time is dominated by waiting
for the slowest participant, not by bytes on the wire.

### 5.5 E5 breakdown

Figures: figures/e5_breakdown.png (allocation of fitting time to local compute,
collectives, centroid update and rank imbalance) and
figures/e5_hybrid_breakdown.png (the same for every hybrid layout). Local
compute is 60-99% of fitting time in every configuration measured here; the
unattributed remainder is below 5% everywhere, so the timers close. Derived
communication volume per rank per iteration is (P-1)/P * (K*D*8 + (K+1)*8)
bytes, i.e. about 3.1 KB at P=8; it is reported as a derived quantity and is
never converted into a time or a bandwidth.

### 5.6 E6 Apple Metal

| N | CPU assign | GPU kernel (B) | dispatch floor | fit A | fit B | accumulate (B) |
|---|---|---|---|---|---|---|
| 20000 | 0.00380 | 0.00215 | 2.8e-04 | 0.0557 | 0.0569 | 0.0091 |
| 50000 | 0.00958 | 0.00340 | 2.5e-04 | 0.1135 | 0.1181 | 0.0230 |
| 100000 | 0.01894 | 0.00527 | 2.6e-04 | 0.2181 | 0.2145 | 0.0513 |
| 200000 | 0.03887 | 0.00822 | 2.7e-04 | 0.4157 | 0.4324 | 0.1149 |
| 400000 | 0.07924 | 0.01615 | 2.9e-04 | 0.9145 | 0.6728 | 0.1611 |

* Assignment kernel vs CPU assignment pass at 400000 observations: 0.0161 s vs
  0.0792 s, 4.9x. At 20000 it is only 1.8x because the 0.28 ms dispatch-plus-sync
  floor is then 13% of the kernel time. This is why kernel time is compared
  against the CPU assignment pass and never against a whole CPU fit.
* Stage A versus Stage B at 400000: 0.9145 s vs 0.6728 s (1.36x). Stage A's
  avoidable uploads alone cost 0.147 s of the 0.914 s; Stage B's per-iteration
  centroid upload costs 0.0096 s in total. The accounting closes:
  0.485 (kernel) + 0.010 (upload) + 0.161 (accumulate) + 0.002 (readback) =
  0.658 s against 0.673 s measured.
* Correctness in every recorded run: label agreement with the double-precision
  CPU reference is 1.000000, centroid maximum absolute difference 0, relative
  inertia difference 0. The float32 kernel only decides assignment; the
  accumulation and the centroid update stay in double on the CPU, so when the
  labels agree the centroids are bitwise identical. The residual risk is a
  near-tie assignment flipping on a different dataset, which is why agreement
  is reported per row instead of being asserted.
* The remaining bottleneck is the CPU accumulation pass (0.161 s of 0.673 s at
  400000, 24%), not the kernel. GPU-side accumulation is the next step and is
  P2.

## 6. Limitations and how each one was handled

1. **E2/E3 points above 8 ranks do not exist yet.** Routed to the HPC kit with a
   mandatory 2-rank checkpoint before production submission. Not presented as
   done anywhere in this repository.
2. **8-core laptop contaminates the 8-rank point.** Documented in
   docs/EXPERIMENTS.md and here; the figure shows all four points, and the text
   explains why 8 ranks is oversubscribed. The pilot pass that was disturbed by
   concurrent work was re-run cleanly and the contaminated rows are quarantined
   in results/pilot/ rather than deleted.
3. **MPI collective time is not network time.** Enforced as an interpretation
   rule in the manifest, and illustrated by the non-monotone hybrid collective
   series. No bandwidth figure is derived.
4. **--rows re-standardizes per subset.** Documented; the E1 slope is a scaling
   estimate for the K-means cost model, not a claim that normalization was held
   fixed across N.
5. **GPU float32.** Exposure is confined to assignment decisions; validated as
   above; agreement is recorded per row, not asserted globally.
6. **Vanda/Atlas scripts are unverified.** They exist with explicit CHECK_ME
   guards, a probe script and a documented checkpoint. Their scheduler syntax is
   the generic PBS Pro / SLURM form, and no module name, queue name, node size
   or queue wait is invented anywhere.
7. **252 MB source broadcast at 256 ranks.** Both distribution paths are
   implemented and measured at small rank counts; the HPC jobs default to the
   cached binary and the log template asks for the parallel-filesystem behaviour
   to be recorded.
8. **No cluster runtime for the report.** Unavoidable without an allocation; the
   status is "blocked by unavailable access", not "done".

## 7. What remains before the deadline

Priority order:

1. Obtain an Atlas or Vanda allocation, run jobs/probe_cluster.sh, fill
   jobs/cluster.env, and pass the 2-rank checkpoint. Then run E1 (cluster
   serial), E2 strong, E3 weak, the 72-core hybrid layouts and one multi-node
   hybrid layout. This is the only outstanding P0 item.
2. Copy the cluster CSVs into results/, rerun scripts/analyze.py, and extend the
   report with the cluster figures.
3. Confirm with the professor whether the Metal implementation satisfies the
   GPU-testing expectation in the assignment.
4. P1 leftovers: none in code - the Metal kernel, plots and metadata are done.
5. P2 (only if time remains): GPU-side accumulation, MPI reduction variants,
   OpenMP schedule comparison.

## 8. Reproduction

    make                       # clean build, no warnings
    make test                  # three suites, all passing
    python3 scripts/smoke_test.py
    python3 scripts/run_e1.py --repetitions 3
    python3 scripts/run_mpi.py --experiment strong --ranks 1,2,4,8 --repetitions 3
    python3 scripts/run_mpi.py --experiment weak --ranks 1,2,4,8 --per-rank 50000 --repetitions 3
    python3 scripts/run_mpi.py --experiment hybrid --ranks 1,2,4 --threads 1,2,4 --max-workers 8 --repetitions 3
    python3 scripts/run_metal.py --rows 20000,50000,100000,200000,400000 --repetitions 3
    python3 scripts/analyze.py

## 2026-11-14 repair and redesign

The 2026-11-05 status above is historical. On 2026-11-14 the project was
audited again after the teammate CUDA merge and the failed Vanda hybrid jobs.

Confirmed and fixed:

1. **Fixed-total resampling was absent from `main`.** It was recovered from
   `stash@{0}` and applied on branch `repair-2026-11-14`. `--resampled-total N`
   now works in serial, OpenMP, MPI and hybrid; MPI/hybrid generate only their
   own partitions; all rank counts use the same source, mapping and initial
   centroids.
2. **Hybrid multi-node launch hung.** `ppr:R:node:pe=T --bind-to core` was
   replaced by `ppr:R:node --bind-to none`, `OMP_PROC_BIND=false`, an exclusive
   allocation, a one-iteration preflight and a hard timeout. The runner now
   verifies `total_workers = ranks × threads`.
3. **PBS metadata was inconsistent.** Allocation `ompthreads` is 1; per-run
   `OMP_NUM_THREADS` is set by the runner. No layout exceeds
   `ncpus/node × nodes`.
4. **Stale binaries.** `common_harry.sh` does `make -B cpu`, writes
   `BUILD_INFO.md5`/`BUILD_INFO.txt`, and production jobs verify the checksums.
5. **Monolithic jobs.** E0/E2/E5/E6-64/E6-128/E6-256/E7 are independent PBS
   jobs; E6 has per-layout preflight/timeout so one hang cannot destroy the
   campaign.
6. **Wrong account/project.** The teammate `personal-e0550588` scripts are not
   submitted; the harry kit uses `personal-harry_chen` and
   `/scratch/harry_chen/DSA5208_Proj3`.
7. **Overflow and CLI edges.** `generated_total`/`materialise_indices` have
   overflow guards; serial/OpenMP reject the MPI-only `--weak-per-rank`.
8. **OpenMP runner and real-data analysis** were added (`run_openmp.py`,
   `analyze_covertype.py`).

Measured data retained:

* E1 39 rows (serial, original + 1M/5M/10M resampled).
* E3 large 10M 27 rows, 133× / 52% at 256 ranks, one hash.
* E4 weak 27 rows on 5 heterogeneous nodes, 55.4% at 256.
* E2 original legacy 27 rows on 5 heterogeneous nodes, same hash as the
  homogeneous 4×64 and 8×32 runs.
* E9 converged run, 46 iterations, inertia 3,722,934.178255, ARI 0.0191.
* CUDA 30 rows, Metal 30 rows.

The old hybrid rows (1554129/1554130) are classified INVALID because of a 40×
spread in the same layout; they are retained as evidence, not used as scaling
results. The repaired E0–E7 campaign is gated on E0 and is not claimed complete
until its CSVs pass `docs/RESULT_VALIDATION.md`.
