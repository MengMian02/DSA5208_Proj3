# Experiment manifest (redesigned 2026-11-14)

Single source of truth for the E0–E9 campaign. Status values:

* **COMPLETE** — expected files/rows exist and passed validation.
* **COMPLETE_LEGACY** — valid measured data from an earlier allocation, kept
  for comparison but superseded by a homogeneous rerun where relevant.
* **QUEUED / RUNNING / FAILED / TIMED_OUT / INVALID / NOT_SUBMITTED** — as
  defined in the task specification.

Primary constants unless stated otherwise: K=7, D=54, seed=42,
`--fixed-iterations`, 30 iterations, 3 repetitions, median reported.

Datasets:

| Name | N | Purpose |
|---|---:|---|
| original Covertype | 581,012 | real data, correctness, small-workload scaling |
| systematic subset | 50k–581,012 | E1 serial N sweep |
| fixed-total resampled | 1M/5M/10M | E1/E2-large/E3/E5/E6 |
| weak resampled | 50,000 × ranks | E4, up to 12.8M at 256 ranks |

The resampled data are deterministic benchmark expansions of real Covertype
rows, not independent observations.

## Status matrix

| ID | Experiment | Configurations | Expected rows | Status | Output |
|---|---|---|---:|---|---|
| E0 | Build/correctness gate | serial/OpenMP/MPI/hybrid; 20k; 200003 resampled; 2-node placement; 5 hybrid mappings (ppr/node/slot/none/core) | pass/fail | **NOT_SUBMITTED** | `logs/e0_<jobid>.log`, `BUILD_INFO.*` |
| E1 | Serial N scaling | 50k/100k/200k/400k/581012 + 1M/5M/10M; K=7 (+K sweep) | 39 | **COMPLETE** | `results/vanda_harry/1552578..._e1_serial.csv` |
| E2 | Original MPI strong | 1,2,4,8,16,32,64,128,256 ranks; 581012 | 27 | **NOT_SUBMITTED** (legacy hetero 27 rows COMPLETE_LEGACY) | `..._e2_strong.csv` |
| E3 | Large MPI strong | same ranks; 10,000,000 resampled | 27 | **COMPLETE** | `..._e2_large.csv` (1552703) |
| E4 | MPI weak | same ranks; 50,000/rank | 27 | **COMPLETE_LEGACY** | `..._e3_weak_256_hetero.csv` (1552672) |
| E5 | OpenMP thread scaling | 1,2,4,8,16,32,64 threads; 581012 and 10M | 21+21 | **NOT_SUBMITTED** | `..._e5_openmp_original.csv`, `..._e5_openmp_10m.csv` |
| E6 | Hybrid 256 | 256x1..4x64 on 4×64 | 21 | **NOT_SUBMITTED** (old rows INVALID) | `..._e6_hybrid_256.csv` |
| E6 | Hybrid 128 | 128x1..4x32 on 2×64 | 18 | **NOT_SUBMITTED** (1552704 TIMED_OUT) | `..._e6_hybrid_128.csv` |
| E6 | Hybrid 64 | 64x1..1x64 on 1×64 | 21 | **NOT_SUBMITTED** | `..._e6_hybrid_64.csv` |
| E7 | Node layout | 8×32; core/ppr32/ppr16 mappings at 128/256 | 10 | **NOT_SUBMITTED** | `..._e7_*.csv` |
| E8 | GPU | CUDA 30 rows; Metal 30 rows | 30+30 | **COMPLETE** | `results/vanda/e7_cuda_1554475.csv`, `results/e6_metal.csv` |
| E9 | Real-data analysis | converged K-means, 581012 | 1 run | **COMPLETE** | `covtype_converged_1552669...`, `covertype_metrics.json` |

## E0 — gate

Purpose: rebuild from source, record compiler/MPI/flags/checksums, verify all
CPU implementations agree, verify fixed-total and weak identities, and prove
that a multi-node hybrid launch starts with the expected worker count.

Pass criteria:

* Build info exists and checksums verify.
* `label_hash` equal for serial/OpenMP 4t/MPI 2,4/hybrid on 20k.
* `label_hash` equal for serial/OpenMP/MPI 1,2,4/hybrid on `--resampled-total
  200003`.
* `1×20000 == 4×5000` weak hash.
* Rank placement spans exactly 2 hosts.
* At least one hybrid mapping gives `total_workers=8` and the reference hash.

Commands are in `jobs/vanda/e0_gate_hc.pbs`. Production jobs use
`-W depend=afterok:<E0 id>`.

## E1 — serial

Already measured by job 1552578 on IN-001: 39 rows, 30 iterations, medians in
`docs/RESULT_VALIDATION.md`. Do not rerun unless the binary or preprocessing
changes. The fit exponent for the resampled N sweep is 1.00.

## E2 — original MPI strong

`e2_strong_hc.pbs`: 4 nodes ×64 cores, `select=4:ncpus=64:mpiprocs=64:
ompthreads=1`, place=scatter, `--binding core` (MPI_FLAGS `--map-by core
--bind-to core`), 1..256 ranks, 3 reps, expected 27 rows. The 1552672 hetero
result remains as a complete legacy record but is not mixed into the
homogeneous E2 curve.

## E3 — large MPI strong (10M)

Already measured by the E2 portion of job 1552703 before its E4 phase
timed out: 27 rows, all rank counts same hash, speedup 133× at 256, efficiency
52%. Because the PBS job exit is -29 but the E2 CSV is complete and internally
consistent, the E2 rows are accepted and the E4 failure is reported separately.

## E4 — MPI weak

Job 1552672 produced 27 rows on a 5-node heterogeneous 256-core allocation;
each rank owns 50,000 observations. It is a valid weak-scaling record
(55.4% efficiency at 256) but the allocation is heterogeneous. The committed
4×64 and 8×32 teammate jobs report 80.2% and 35.8%; those are separate
homogeneous records and are not merged with this one.

## E5 — OpenMP

`e5_openmp_hc.pbs`: one 64-core node, threads 1..64, original 581012 and 10M
resampled, 3 reps each, 21 rows per dataset. `run_openmp.py` sets and verifies
`OMP_NUM_THREADS`; `OMP_PLACES=cores` and `OMP_PROC_BIND=close`.

## E6 — hybrid

Independent jobs at 64, 128 and 256 cores, all on the fixed 10M workload.
Every layout is `ranks × threads = budget`; per-node cores are budget/nodes.
The launch uses the mapping certified by E0 (`HYBRID_LAUNCH`; normally
`--map-by ppr:R:node --bind-to none`), with `OMP_PROC_BIND=false` and
`OMP_PLACES` unset. Each layout first runs one 20k-observation iteration with
a 90 s timeout; a failed preflight is recorded and the layout is skipped. Timed
runs use a 180 s timeout. Expected rows 21/18/21.

## E7 — node layout

`e7_layout_hc.pbs`: one 8×32 CN allocation. It runs 128/256 ranks with
`--map-by core`, `--map-by ppr:32:node`, and 128 ranks with
`--map-by ppr:16:node`, printing per-host rank counts. The existing 4×64 GN
data are compared only as a hardware-confounded reference.

## E8 — GPU

Preserve and validate existing results; no new GPU job unless a metric is
missing. CUDA: A40, 30 rows, label agreement ≥0.9999875. Metal: M1 Pro,
30 rows, agreement 1.0. GPU kernel-only time is never compared with CPU
end-to-end time.

## E9 — real data

Converged serial run (46 iterations, inertia 3,722,934.178255) and
`scripts/analyze_covertype.py`: cluster sizes, centroid profiles, contingency
table, ARI=0.0191, NMI=0.0562. Only the original 581,012 observations are used.

## Expected-results manifest

`results/vanda_harry/expected_results.csv` lists every planned job, its
configuration count, expected CSV rows, output path and dependency. It is the
checklist used by `docs/RESULT_VALIDATION.md`.

## Reproduction

```
ssh harry_chen@vanda.nus.edu.sg
cd /scratch/harry_chen/DSA5208_Proj3
qsub -P personal-harry_chen jobs/vanda/e0_gate_hc.pbs
# after E0 passes:
qsub -P personal-harry_chen -W depend=afterok:<E0> jobs/vanda/e2_strong_hc.pbs
qsub -P personal-harry_chen -W depend=afterok:<E0> jobs/vanda/e5_openmp_hc.pbs
qsub -P personal-harry_chen -W depend=afterok:<E0> jobs/vanda/e6_hybrid_64_hc.pbs
qsub -P personal-harry_chen -W depend=afterok:<E0> jobs/vanda/e6_hybrid_128_hc.pbs
qsub -P personal-harry_chen -W depend=afterok:<E0> jobs/vanda/e6_hybrid_256_hc.pbs
qsub -P personal-harry_chen -W depend=afterok:<E0> jobs/vanda/e7_layout_hc.pbs
```
