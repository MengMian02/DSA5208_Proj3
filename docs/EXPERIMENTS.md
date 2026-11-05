# Experiment manifest

Single source of truth for what each experiment varies, what is held constant,
which metric is primary, and what has actually been measured. Status values:

* **measured locally** - executed on this machine; rows exist in results/.
* **instrumented, awaiting HPC** - code and scripts exist, no cluster data yet.
* **not run** - nothing executed, nothing claimed.

Hardware used for every local record: Apple M1 Pro, 8 cores (6 performance + 2
efficiency), 16 GB unified memory, macOS, Apple clang 21, OpenMPI 5.0.2,
-O3 -std=c++17 -DNDEBUG. Host name and compiler string are stored per row.

Common constants unless an experiment says otherwise: D = 54, seed = 42,
stopping mode = fixed_iterations, repetitions = 3.

| Exp | Varies | Held constant | Primary metric | Status |
|---|---|---|---|---|
| E1A | N in 50k, 100k, 200k, 400k, 581012 | K = 7, I = 30 | fit_s | measured locally |
| E1B | K in 2, 4, 7, 10, 16 | N = 581012, I = 30 | fit_s | measured locally |
| E2 | 1, 4, 16, 32, 64, 128, 256 MPI ranks | full Covertype, K = 7, I = 30 | fit_s, speedup vs 1-rank MPI | ranks 1-8 measured locally; 16-256 awaiting HPC |
| E3 | 1, 4, 16, 32, 64, 128, 256 MPI ranks | 50000 generated observations per rank, K = 7, I = 30 | fit_s, weak efficiency | ranks 1-8 measured locally; 16-256 awaiting HPC |
| E4 | ranks x threads at equal cores | K = 7, I = 30, full Covertype | fit_s at equal core count | 1x1,1x2,1x4,2x1,2x2,2x4,4x1,4x2 measured locally; 72-core and multi-node layouts awaiting HPC |
| E5 | rank count (reuses E2/E3/E4 runs) | as E2/E3/E4 | local_compute_s, mpi_reduction_s, their shares of fit_s | measured locally for ranks 1-8 |
| E6 | N in 20k..400k, stage A vs B | K = 7, I = 30 | CPU vs GPU assignment time, GPU-assisted fit time | measured locally on M1 Pro GPU |

## E1 - serial scaling

Objective: how serial fitting time grows with observations and clusters,
compared with the expected O(I * N * K * D).

Design notes that matter for reading the numbers:

* Each N is the evenly spaced --rows subset, so the class proportions of the
  class-ordered Covertype file are preserved.
* Preprocessing statistics are recomputed on each subset; the subset is the
  dataset for that configuration.
* fit_s excludes load, preprocess, k-means++ and validation.
* The fitted log-log slope is annotated on the figure. For E1A the slope is
  close to 1; for E1B it is below 1 because the per-iteration observation sum
  (N * D work) does not grow with K, so K = 16 amortises that fixed cost.

## E2 - MPI strong scaling

Observation-wise decomposition. Each rank owns a contiguous block of
observations, computes local cluster sums and counts, and participates in two
MPI_Allreduce calls per iteration (K x D doubles, then K + 1 uint64 including
the changed-label count). Partitions tolerate any rank count because rows are
split as base = n / P with the first n % P ranks taking one extra row.

The 1-rank MPI run is the primary baseline (it includes the same distribution
and collective path as the rest). The serial program is kept as a separate
reference and is never used as the MPI speedup denominator.

speedup(P) = T_fit(1) / T_fit(P); efficiency(P) = speedup(P) / P.

Local caveat that must be stated in the report: this machine has 8 cores of two
different kinds, so 8 ranks already oversubscribe the performance cores and the
8-rank point is not a clean strong-scaling measurement. The rank sweep up to 256
belongs to the cluster.

## E3 - MPI weak scaling

50000 generated observations per rank, so the total grows 50000 -> 12.8 M.

E_weak(P) = T_fit(1, N0) / T_fit(P, P * N0).

Design decisions:

* The logical dataset is produced in memory by the documented global-index hash
  (see README). No expanded file is written.
* Initial centroids are k-means++ on the source Covertype matrix, identical for
  every rank count. A different k-means++ per expanded configuration is
  explicitly not used.
* Source distribution is measured separately (source_bcast_s / load_s for the
  cached binary path) and excluded from fit_s.
* The workload is an expanded benchmark built by resampling real Covertype rows.
  No statistical conclusion about Covertype itself may be drawn from E3.

## E4 - hybrid MPI + OpenMP

Same decomposition as E2 with an OpenMP team inside every rank: thread-local
sum and count arrays, a fixed-order fold into rank-local totals, then MPI
reductions, so the result does not depend on thread scheduling. MPI calls only
the main thread (MPI_THREAD_FUNNELED, checked at start-up).

Comparisons are only made at equal core count. The 72-core sweep prepared for
the cluster is 72x1, 36x2, 18x4, 9x8, 4x18, 1x72; on this 8-core machine the
equivalent local sweep is 8x1, 4x2, 2x4, 1x8. The question is whether fewer
ranks (less collective traffic) compensate for the OpenMP fold and barrier.

## E5 - computation and communication breakdown

Reuses the E2/E3/E4 runs; no separate benchmark. Reported phases per iteration:
assignment+accumulation, OpenMP fold, MPI collectives, centroid update, and the
unattributed remainder of fit_s.

Interpretation rules fixed in advance:

* mpi_reduction_s is collective latency, which includes waiting for the slowest
  rank. It is not labelled network transfer, and no bandwidth is derived from it.
* fit_imbalance_s = max fit - min fit across ranks is the load-imbalance
  indicator. It only covers whole-rank skew, not intra-rank thread skew.
* Communication volume is reported as a derived quantity (ranks - 1) / ranks x
  (K*D*8 + (K+1)*8) bytes per rank per iteration, clearly separated from
  measured times.

## E6 - Apple Metal

Local-only. No CUDA, no HPC GPU scripts.

* Stage A (--mode assign): naive per-iteration transfer of the observation
  matrix and the centroids, explicit label readback, CPU accumulation.
* Stage B (--mode full): observation matrix resident on the GPU, only K x D
  centroids re-uploaded per iteration, labels read through unified shared
  memory.
* Precision: Metal on Apple GPUs has no IEEE double, so the kernel is float.
  The CPU reference stays double. Comparison is by label agreement, centroid
  maximum absolute difference and relative inertia difference - never bitwise.
* The GPU assignment kernel time is compared with a CPU assignment pass of the
  same work in the same run, and end-to-end fitting times are compared
  separately. A kernel/whole-fit ratio is never presented as speedup.

## Result file schema

results/<experiment>.csv, one row per repetition (never overwritten, appended):

experiment, timestamp, run_id, repetition, implementation, host, cpu_model,
cores, dataset, rows, features, k, max_iter, iterations, stopping_mode,
converged, ranks, threads, total_workers, weak, obs_per_rank, fit_s, fit_min_s,
fit_imbalance_s, load_s, preprocess_s, init_s, distribution_s, mpi_reduction_s,
local_compute_s, thread_reduction_s, centroid_update_s, validation_s, total_s,
sse, label_hash, compiler, compiler_flags, mpi_version, notes

results/e6_metal.csv additionally carries stage, device, cpu_assign_s,
gpu_assign_s, gpu_kernel_s, gpu_kernel_total_s, gpu_host_total_s,
dispatch_floor_s, upload_s, readback_s, accumulate_s, centroid_pack_s,
label_agreement, centroid_max_abs_diff, inertia_rel_diff.

Aggregation: median over repetitions for every plotted point, with the min/max
range drawn as the error bar. results/SUMMARY.md holds the plain tables.

## Exact commands used for the local records

    python3 scripts/run_e1.py --repetitions 3
    python3 scripts/run_mpi.py --experiment strong --ranks 1,2,4,8 --repetitions 3
    python3 scripts/run_mpi.py --experiment weak   --ranks 1,2,4,8 --per-rank 50000 --repetitions 3
    python3 scripts/run_mpi.py --experiment hybrid --ranks 1,2,4 --threads 1,2,4 --max-workers 8 --repetitions 3
    python3 scripts/run_metal.py --rows 20000,50000,100000,200000,400000 --repetitions 3
    python3 scripts/analyze.py

The cluster counterparts of these commands live in jobs/atlas/*.pbs and
jobs/vanda/*.slurm and have **not** been submitted.
