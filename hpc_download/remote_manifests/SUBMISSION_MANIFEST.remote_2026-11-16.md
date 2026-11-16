# SUBMISSION_MANIFEST — Vanda account `harry_chen`

Updated 2026-11-15 13:08 SGT. Workspace
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

## 2. Repaired campaign — submitted 2026-11-15 13:07-13:08 SGT

PBS recovered on 2026-11-15. `qstat -B` returned normally, `qstat -u
harry_chen` was empty, and `qstat -x` showed no E0–E7 job from the failed
attempt. E0 was submitted first; the other six jobs were then accepted with
`-W depend=afterok:1557208`. All seven IDs were verified with `qstat -xf`.

Snapshot at 13:08: E0 `R`, six dependents `H` (held by the `afterok` dependency).

| ID | Experiment | PBS Job ID | Request | Dataset | Expected rows | Dependency | Status |
|---|---|---|---|---|---:|---|---|
| E0 | build + correctness gate | **1557208** | 2 nodes × 8 cores, 25 min, `auto_free` | 20k + 200003 | pass/fail | none | R |
| E2 | original MPI strong | **1557218** | 4 × 64 cores, 25 min, `auto` | 581,012 | 27 | afterok:1557208 | H |
| E5 | OpenMP scaling | **1557219** | 1 × 64 cores, 40 min, `auto_free` | 581,012 + 10M | 21+21 | afterok:1557208 | H |
| E6-64 | hybrid | **1557220** | 1 × 64 cores, 40 min, `auto_free` | 10M | 21 | afterok:1557208 | H |
| E6-128 | hybrid | **1557221** | 2 × 64 cores, 35 min, `auto_free` | 10M | 18 | afterok:1557208 | H |
| E6-256 | hybrid | **1557222** | 4 × 64 cores, 45 min, `auto` | 10M | 21 | afterok:1557208 | H |
| E7 | node layout | **1557223** | 8 × 32 cores, 25 min, `auto` | 581,012 | 10 | afterok:1557208 | H |

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
* No production job is released without E0 success (`afterok` dependency).
* E0 certifies the hybrid mapping in `HYBRID_LAUNCH`; E6 reads that file.
* No `pe=` mapping is used; balanced `ppr`/`node`/`slot` mappings with
  `--bind-to none` are tested first.
* Every layout has a preflight and a hard timeout.
* `BUILD_INFO.md5` must verify before production.
* No result CSV is overwritten; every row carries the job ID.
