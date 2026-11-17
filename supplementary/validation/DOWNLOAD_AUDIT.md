# DOWNLOAD_AUDIT — Vanda campaign

Downloaded 2026-11-16 from
the supplementary Vanda results directory and the job logs.
Local root: `DSA5208_Proj3_Harry/results/vanda_harry/`; logs:
`DSA5208_Proj3_Harry/hpc_download/logs/`; build certificate:
`hpc_download/build/`.

Status snapshot from the Vanda `qstat` history at approximately 13:16 SGT and
`qstat -xf` for each job. Exit codes are PBS `Exit_status`. Actual walltime is
computed from `stime`/`mtime`; it is not the requested walltime.

## Completeness table

| Experiment | Job ID | PBS status | Exit | Requested wall | Actual walltime | Expected rows | Actual rows | Completeness | Local result |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| E0 build/correctness | 1557208 | F | 0 | 00:25 | 00:01:14 | pass/fail | PASS | **COMPLETE** | `hpc_download/logs/km-hc-e0gate.o1557208`, `hpc_download/build/` |
| E1 serial | 1552578 | F | 0 | 01:30 | 00:06:27 | 39 | 39 | **COMPLETE** | `results/vanda_harry/1552578..._e1_serial.csv` |
| E2 strong | 1557218 | F | **1** | 00:25 | 00:07:09 | 27 | **24** | **PARTIAL** | `results/vanda_harry/1557218..._e2_strong.csv` |
| E2 1-rank recovery | 1563234 | F | 0 | 00:15 | 00:00:20 | 3 | 3 | **COMPLETE, standalone** | `results/vanda_harry/1563234..._e2_rank1_recovery.csv` |
| E3 large 10M | 1552703 | F | -29 | 00:30 | 30:29 job; E3 section ~6 min | 27 | 27 | **COMPLETE for E3** | `results/vanda_harry/1552703..._e2_large.csv` |
| E4 weak | 1552672 | F | 0 | 00:25 | 00:03:24 | 27 | 27 | **COMPLETE** | `results/vanda_harry/1552672..._e3_weak_256_hetero.csv` |
| E5 OpenMP original | 1557219 | F | 0 | 00:40 | 00:07:23 | 21 | 21 | **COMPLETE** | `results/vanda_harry/1557219..._e5_openmp_original.csv` |
| E5 OpenMP 10M | 1557219 | F | 0 | 00:40 | 00:07:23 | 21 | 21 | **COMPLETE** | `results/vanda_harry/1557219..._e5_openmp_10m.csv` |
| E6-64 Hybrid | 1557220 | F | 0 | 00:40 | 00:02:10 | 21 | 21 | **COMPLETE** | `results/vanda_harry/1557220..._e6_hybrid_64.csv` |
| E6-128 Hybrid | 1557221 | **Q** | — | 00:35 | 0 | 18 | 0 | **QUEUED** | none |
| E6-256 Hybrid | 1557222 | F | **1** | 00:45 | 00:03:16 | 21 | **18** | **PARTIAL** | `results/vanda_harry/1557222..._e6_hybrid_256.csv` |
| E6-256 256×1 recovery | 1563235 | F | 0 | 00:30 | 00:00:18 | 3 | 3 | **COMPLETE, separate allocation** | `results/vanda_harry/1563235..._e6_256_rank1_recovery.csv` |
| E7 layout | 1557223 | F | 0 | 00:25 | 00:01:06 | 10 | 10 | **COMPLETE** | `1557223..._e7_{core,ppr32,ppr16}.csv` |
| E9 Covertype | 1552669 | F | 0 | 00:30 | 00:00:08 | 1 run | 1 | **COMPLETE** | `covtype_converged_1552669.../` |

## Partial runs

### E2 recovery 1563234 (complete, 3/3)
The single-node `--binding none` recovery completed three rows on IN-010 with
median 3.429 s and the same label hash. It is standalone evidence only: the
node family differs from the original CN/GN allocation, so it is not inserted
into the 4×64 speedup curve.

### E2 1557218 (exit 1, 24/27)
`km-hc-e2strong.o1557218` records a 300 s timeout for the `ranks=1` run before
any rank output; all ranks 2--256 completed three repetitions each. The 24
rows have one label hash and the expected `rows=581012`, `K=7`, `D=54`,
`iterations=30`. The missing `1`-rank baseline means this particular job cannot
by itself produce a speedup curve; the legacy complete E2 result (1552672,
27 rows) is retained separately. **Do not treat 1557218 as a complete strong
scaling campaign.**

### E6-256 1557222 (exit 1, 18/21)
`km-hc-e6-256.o1557222` shows the `256x1` preflight timing out at 90 s; the
six remaining layouts (128x2, 64x4, 32x8, 16x16, 8x32, 4x64) completed three
repetitions each. The 18 rows have one label hash and `rows=581012`; they are
valid for the multi-thread layouts but **not** for the missing 256x1 layout.
This run also used a mixed CN/GN allocation and the unpinned mapping; treat it
as a placement-limited partial result.

### E6-256 recovery 1563235 (complete, 3/3)
The four-CN-node pinned recovery (`--map-by core --bind-to core`) completed
three 256×1 rows: median 0.32014 s, label hash `14076111245358280480`, SSE
`64089789.818295`. It is a separate allocation from the six unpinned
multi-thread layouts, so it completes the measured set but does not turn the
whole comparison into a same-binding experiment.

### E6-128 1557221 (queued)
No log or CSV exists. It was still `Q` with
`Not Running: Insufficient amount of resource: node_pool` at the snapshot.
No result may be claimed.

## Completed and usable

* E0: PASSED. The log records the forced build, all serial/OpenMP/MPI/hybrid
  hashes, fixed-total resampling equality, weak identity, two-host placement,
  and `HYBRID_MODE=ppr-none`. Build checksums are in
  `hpc_download/build/BUILD_INFO.*`.
* E1: 39 rows, 3 repetitions each for 5 N values, 5 K values and 3 resampled
  sizes.
* E3 large: 27 rows, 10M logical observations, one hash across rank counts.
* E4 weak: 27 rows, 50,000 observations/rank.
* E5: 2 × 21 rows (original and 10M), 1--64 threads, three repetitions.
* E6-64: 21 rows, seven layouts at 64 cores, three repetitions.
* E7: 10 rows across `core`, `ppr32`, `ppr16`; `ppr16` placement log shows
  16 ranks on each of eight hosts. The `core`/`ppr32` placement printouts are
  absent from the log, but their CSVs are complete; placement interpretation
  is therefore limited.
* E9: converged original-data run and centroids.

## Checksums

SHA-256 for all downloaded results and logs is in
`hpc_download/checksums.sha256` (45 entries). The remote staging tree contained
16 files; 8 were new locally, 8 were byte-identical, and 0 conflicted. The local
result tree contains 27 files because it also holds the pre-existing analysis
artefacts. Nineteen logs/build files were new; 0 logs conflicted. No existing
local file was deleted. The original remote `SUBMISSION_MANIFEST.md` and
`expected_results.csv` are preserved in `hpc_download/remote_manifests/` with
SHA-256 sidecars, because the local copies were updated with the downloaded
outcomes.

## Invalid / do not use as performance evidence

* The earlier unpinned 256-core Hybrid rows (`1554129`, `1554130`) remain
  invalid and are not part of this download campaign.
* E2 1557218 without its 1-rank baseline is PARTIAL, not a complete curve.
* E6-256 1557222 without 256x1 is PARTIAL.
* E6-128 1557221 is QUEUED, no output.
