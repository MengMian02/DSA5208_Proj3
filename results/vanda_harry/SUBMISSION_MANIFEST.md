# SUBMISSION_MANIFEST — Vanda account `harry_chen`

Updated 2026-11-14 18:20 SGT. Workspace
`/scratch/harry_chen/DSA5208_Proj3`; project `personal-harry_chen`.
`Q` = queued, `F` = finished, `—` = not submitted. A PBS job is never marked
complete unless its CSV/log was inspected.

## 1. Previous jobs (measured before this repair)

| Job ID | Experiment | Request | Dataset | Status | Evidence |
|---|---|---|---|---|---|
| 1552574 | E5 build/correctness gate | 1×12 cores, 30 min | 20k + 200003 resampled | **F, exit 0** | `km-hc-smoke.o1552574` |
| 1552575 | two-node MPI gate | 2×4 cores, 20 min | 2,000 | **F, exit 0** | `km-hc-diag2.o1552575` |
| 1552578 | E1 serial | 1 core, 90 min | 50k–10M | **F, exit 0** | `1552578..._e1_serial.csv`, 39 rows |
| 1552669 | E9 converged analysis | 1 core, 30 min | 581,012 | **F, exit 0** | `covtype_converged_1552669...` |
| 1552672 | E2 original + E4 weak | 256 cores, 9 nodes hetero, 25 min | 581,012 / 50k per rank | **F, exit 0** | 27 + 27 rows |
| 1552703 | E3 large 10M + E6-256 | 256 cores, 4×64, 30 min | 10M | **E3 complete; E6 TIMED_OUT** | `..._e2_large.csv` 27 rows; E6 no CSV; exit -29 |
| 1552704 | E6-128 | 128 cores, 2×64, 20 min | 10M | **TIMED_OUT** | no CSV, zero CPU, exit -29 |

The 1552703 E3 rows are valid despite the job exit: the E2 section finished
with 27/27 rows and all rank counts share hash `14076111245358280480`; the E4
section then hung and consumed the remaining walltime.

## 2. Repaired campaign (submission IDs to be filled after PBS recovers)

At 18:05–18:20 the PBS server port 15001 was closed and `qstat` timed out. Two
hung `qsub` processes were killed; no duplicate job was created. The following
jobs are prepared and will be submitted only when `qstat -u harry_chen` returns.

| ID | Experiment | PBS Job ID | Request | Dataset | Expected rows | Dependency | Status |
|---|---|---|---|---|---:|---|---|
| E0 | build + correctness gate | TBD | 2×8 cores, 25 min | 20k + 200003 | pass/fail | none | NOT_SUBMITTED |
| E2 | original MPI strong | TBD | 4×64, 25 min | 581,012 | 27 | E0 | NOT_SUBMITTED |
| E5 | OpenMP scaling | TBD | 1×64, 40 min | 581,012 + 10M | 21+21 | E0 | NOT_SUBMITTED |
| E6-64 | hybrid | TBD | 1×64, 30 min | 10M | 21 | E0 | NOT_SUBMITTED |
| E6-128 | hybrid | TBD | 2×64, 35 min | 10M | 18 | E0 | NOT_SUBMITTED |
| E6-256 | hybrid | TBD | 4×64, 45 min | 10M | 21 | E0 | NOT_SUBMITTED |
| E7 | node layout | TBD | 8×32, 25 min | 581,012 | 10 | E0 | NOT_SUBMITTED |

E3 large and E4 weak are already complete from 1552703 and 1552672; no duplicate
job is submitted for them. E1 and E9 are complete. E8 GPU results are preserved
(CUDA 30 rows, Metal 30 rows); no new GPU job is submitted.

## 3. Output paths

All outputs go under `/scratch/harry_chen/DSA5208_Proj3/results/vanda_harry/`
with the full PBS job ID in the filename, e.g.
`<jobid>.stdct-mgmt-02_e2_strong.csv`. Logs are
`/scratch/harry_chen/DSA5208_Proj3/<jobname>.o<jobid>`.

## 4. Rules enforced

* Only `personal-harry_chen` and `/scratch/harry_chen` are used.
* No job is submitted without E0 success (`afterok` dependency).
* No `pe=` hybrid mapping; `ppr:R:node` + `--bind-to none` is tested first.
* Every layout has a preflight and a hard timeout.
* `BUILD_INFO.md5` must verify before production.
* No result CSV is overwritten; every row carries the job ID.
