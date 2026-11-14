# RESULT_VALIDATION — measured artefacts and status

Date: 2026-11-14. A row count is the number of data rows in the CSV, excluding
the header. `COMPLETE` requires the expected row count, the expected rank/thread
set, identical label hashes where the dataset is fixed, and plausible timings.
A PBS exit status alone is never used as proof of completeness.

## 1. Status matrix

| Experiment | Expected rows | Measured rows | Status | Evidence |
|---|---:|---:|---|---|
| E0 build/correctness gate | no CSV | — | **NOT_SUBMITTED** | PBS unresponsive; scripts ready |
| E1 serial scaling | 39 | 39 | **COMPLETE** | `results/vanda_harry/1552578..._e1_serial.csv` |
| E2 original, homogeneous 4×64 | 27 | 0 | **NOT_SUBMITTED** | new job planned |
| E2 original, legacy 5-node hetero | 27 | 27 | **COMPLETE_LEGACY** | `1552672..._e2_strong_256_hetero.csv` |
| E3 large 10M, 4×64 | 27 | 27 | **COMPLETE** | `1552703..._e2_large.csv` |
| E4 weak 50k/rank, 5-node hetero | 27 | 27 | **COMPLETE_LEGACY** | `1552672..._e3_weak_256_hetero.csv` |
| E5 OpenMP thread scaling | 21 + 21 | 0 | **NOT_SUBMITTED** | `run_openmp.py` ready |
| E6 hybrid 256 | 21 | 21 | **INVALID** | `1554129`, variance 0.027–1.087 s |
| E6 hybrid 128 (teammate 8×32) | 18 | 18 | **INVALID** | `1554130`, 3.4–3.6 s and inconsistent 256×1 |
| E6 hybrid 128 (our 2×64) | 18 | 0 | **FAILED (walltime)** | `1552704`, zero CPU, no CSV |
| E6 hybrid 64/128/256 repaired | 21/18/21 | 0 | **NOT_SUBMITTED** | new jobs planned |
| E7 node-layout diagnostic | 10 | 0 | **NOT_SUBMITTED** | `e7_layout_hc.pbs` ready |
| E8 CUDA | 30 | 30 | **COMPLETE** | `results/vanda/e7_cuda_1554475.csv` |
| E8 Apple Metal | 30 | 30 | **COMPLETE** | `results/e6_metal.csv` |
| E9 real-data converged run | 1 run | 1 | **COMPLETE** | `covtype_converged_1552669...` |

## 2. E1 serial (39 rows)

Rows: E1A 5 sizes ×3, E1B 5 K values ×3, E1C 3 resampled sizes ×3 = 15+15+9
= 39. All `iterations=30`, `ranks=1`, `threads=1`. Host IN-001.

| Dataset | N | K | fit_s median | total_s median | generation_s |
|---|---:|---:|---:|---:|---:|
| original | 50,000 | 7 | 0.3889 | 1.3182 | 0 |
| original | 100,000 | 7 | 0.5460 | 1.2344 | 0 |
| original | 200,000 | 7 | 1.0832 | 1.8342 | 0 |
| original | 400,000 | 7 | 2.1839 | 3.0552 | 0 |
| original | 581,012 | 7 | 3.1541 | 4.1036 | 0 |
| resampled | 1,000,000 | 7 | 5.4203 | 6.5967 | 0.135 |
| resampled | 5,000,000 | 7 | 27.0723 | 29.4136 | 0.668 |
| resampled | 10,000,000 | 7 | 54.2535 | 58.0566 | 1.339 |

Fitted log-log exponent for E1C = 1.00 (O(N)); E1B K exponent = 0.79.
The original N series is not perfectly linear at small N (0.88 fit) because of
cache and fixed per-iteration overhead; this is real measured behaviour, not a
missing row.

## 3. E2 original — legacy 1552672 (27 rows)

5-node heterogeneous allocation: 72+72+32+56+24 cores = 256, host CN-077.
Nine rank counts ×3 reps, all `iterations=30`, all label hashes
`15912193932519828622` (identical to the 4×64 and 8×32 teammate runs).
This is a valid fixed-dataset correctness/performance record, but the node
fragments are heterogeneous; the new homogeneous E2 will supersede it for the
final report.

Median fit: 1 rank 2.884 s, 256 ranks 0.0245 s (117.6×, 45.9% efficiency).
Rep 1 at 1 rank is 3.607 s vs 2.880/2.884 s; median used.

## 4. E3 large 10M — 1552703 (27 rows)

4×64 GN-A40 nodes (host GN-A40-055). Nine rank counts ×3 reps, all
`rows=10000000`, `iterations=30`, dataset `covtype_resampled_10M`, all hashes
`14076111245358280480`. This proves that the fixed-total mapping is independent
of rank count: 1 and 256 ranks produced the same label hash.

| Ranks | fit_s median | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 49.805 | 1.00 | 100% |
| 8 | 6.287 | 7.92 | 99.0% |
| 32 | 1.812 | 27.5 | 85.9% |
| 64 | 1.158 | 43.0 | 67.2% |
| 128 | 0.692 | 71.9 | 56.2% |
| 256 | 0.374 | 133.0 | 52.0% |

The job then entered the E4 hybrid phase and was killed at 1821 s walltime;
the E2 rows above are complete and must not be discarded because the job exit
was -29.

## 5. E4 weak — legacy 1552672 (27 rows)

Nine rank counts ×3 reps, `rows=50000×ranks`, `obs_per_rank=50000`,
`iterations=30`. Label hashes differ by rank count because the logical dataset
legitimately grows with P; this is not a correctness failure.

Weak efficiency at 256 ranks on this 5-node heterogeneous allocation:
1-rank 0.2459 s / 256-rank 0.4440 s = **55.4%**. The committed homogeneous
jobs report 80.2% (4×64, GN-A40) and 35.8% (8×32, CN), respectively; those
baselines are 0.3946 s and 0.3916 s. The three numbers must not be mixed into
one curve.

## 6. E6 hybrid — existing rows are invalid

`results/vanda/vanda_e4_hybrid256_1554129...csv` has 21 rows and
`..._8x32_1554130...csv` has 18 rows, but they fail the plausibility test:

* 256×1: 0.0273, 0.7690, 1.0875 s (40× spread).
* 8×32: 3.42–3.55 s for every repetition, with collective 0.73–0.84 s.
* 4×64/8×32 rows are 50–100× slower than the equivalent MPI-only work.

These are not usable as hybrid scaling results. The repaired jobs add a
one-iteration preflight per layout and a 180 s timeout, and verify
`total_workers = ranks × threads`.

## 7. E8 GPU correctness

### CUDA (NVIDIA A40, 30 rows)

| Stage | Rows | Label agreement | Max centroid diff | Max relative inertia diff |
|---|---:|---:|---:|---:|
| A_assign | 50k–581k | ≥0.9999875 | 8.54e-5 | 3.2e-8 |
| B_full | 50k–581k | ≥0.9999875 | 8.54e-5 | 3.2e-8 |

Only 400k shows the tolerance-level disagreement; 50k/100k/200k/581k are
bitwise-identical labels. The GPU arithmetic is float32 while the reference is
double, so this is expected. Timing boundaries: `gpu_kernel_s` is CUDA-event
kernel time, `upload_s`/`readback_s`/`accumulate_s` are host phases, and
`fit_s` is the full fitting loop. Kernel-only time must not be compared with
CPU end-to-end time.

### Apple Metal (30 rows)

Label agreement 1.0 for every stage/size, 20k–400k. Stage B end-to-end fit
0.673 s at 400k vs 0.915 s for stage A; GPU kernel 0.020 s vs CPU assignment
0.079 s. Metal is float32 and runs on an M1 Pro, not Vanda; it is a separate
platform, not a Vanda scaling point.

## 8. E9 real-data converged run

* 581,012 real observations, K=7, D=54, convergence stopping.
* 46 iterations, inertia **3,722,934.178255**.
* Cluster sizes: 102,132 / 54,851 / 153,387 / 54,961 / 66,308 / 102,343 / 47,030.
* ARI **0.0191**, NMI **0.0562**, AMI **0.0562**.
* Every cluster is majority cover type 2; type 1 is split across all clusters.
* The Python recomputation from the saved centroids reproduces inertia
  3,722,934.178252 (rounding), so the centroid file and the C++ run agree.

## 9. Duplicate/row-integrity checks

* E2 original CSVs: 9 rank values × 3 reps, no duplicate `(ranks, repetition)`.
* E1: 39 unique `(experiment, rows, k, repetition)` keys.
* E3 large: 27 unique `(ranks, repetition)` keys, all rows=10M.
* E4 weak: 27 unique keys, all obs_per_rank=50,000.
* CUDA: 30 unique `(stage, rows, repetition)` keys.
* No result CSV was overwritten; all new output paths include the PBS job ID.

## 10. What remains before a complete report

1. E0 must pass on a compute node and multi-node allocation.
2. E2 homogeneous, E5, E6 repaired, E7 must run and produce their expected row
   counts.
3. E6 layouts with preflight failures must be listed explicitly; they are not
   silently dropped.
4. E7 must print per-host rank counts so the 4×64 vs 8×32 anomaly can be
   classified as hardware-confounded or placement-caused.
