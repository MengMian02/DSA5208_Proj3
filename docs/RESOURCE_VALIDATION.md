# RESOURCE_VALIDATION — Vanda PBS requests

Date: 2026-11-14. Account: `harry_chen`, project `personal-harry_chen`.
Workspace: `/scratch/harry_chen/DSA5208_Proj3`.

Every request below is checked against the limits observed on Vanda. The
scheduler reserves memory proportionally to `ncpus` on `batch_cpu`
(a 4×64 request is recorded as `mem=444GB` per chunk), so a 256-core batch job
needs four nearly empty 72-core nodes. That is a schedulability fact, not an
oversubscription.

## 1. Queue limits observed

| Queue | Max ncpus | Max chunks | Max walltime | Per-user limit | Node pool |
|---|---:|---:|---:|---|---|
| `batch_cpu` (route `auto`) | 720 | 10 | 240 h | 6 jobs, 720 ncpus | `cpu_node` |
| `cpu_parallel` (route `auto_free`) | 160 | 10 | 168 h | 2 jobs, 160 ncpus | `cpu_free` |
| `cpu_serial` (route `auto_free`) | 1 | 1 | 168 h | 32 jobs | `cpu_free` |

* `select=N` means N **chunks**, not N hosts. All multi-node jobs set
  `place=scatter`.
* Multi-node OpenMPI 4.1.8 uses PBS TM only when `/opt/pbs/lib` is on
  `LD_LIBRARY_PATH`; `common_harry.sh` adds it only when the allocation spans
  more than one host.
* `ompthreads` in `#PBS -l select` is allocation metadata. It does **not** set
  `OMP_NUM_THREADS`; every production runner sets and verifies the actual team
  size.

## 2. Requested production jobs

| ID | Job | Queue | select | Nodes | Cores | Walltime | Memory request | Valid? |
|---|---|---|---|---:|---:|---:|---|---|
| E0 | `e0_gate_hc.pbs` | `auto_free` → cpu_parallel | `2:ncpus=8:mpiprocs=8:ompthreads=1:mem=16gb` | 2 | 16 | 00:25 | 16 GB/chunk | yes |
| E2 | `e2_strong_hc.pbs` | `auto` → batch_cpu | `4:ncpus=64:mpiprocs=64:ompthreads=1` | 4 | 256 | 00:25 | site hook 444 GB/chunk | yes |
| E5 | `e5_openmp_hc.pbs` | `auto_free` → cpu_parallel | `1:ncpus=64:mpiprocs=1:ompthreads=1:mem=64gb` | 1 | 64 | 00:40 | 64 GB | yes |
| E6-256 | `e6_hybrid_256_hc.pbs` | `auto` → batch_cpu | `4:ncpus=64:mpiprocs=64:ompthreads=1:mem=100gb` | 4 | 256 | 00:45 | site hook 444 GB/chunk | yes |
| E6-128 | `e6_hybrid_128_hc.pbs` | `auto_free` → cpu_parallel | `2:ncpus=64:mpiprocs=64:ompthreads=1:mem=100gb` | 2 | 128 | 00:35 | 100 GB/chunk | yes |
| E6-64 | `e6_hybrid_64_hc.pbs` | `auto_free` → cpu_parallel | `1:ncpus=64:mpiprocs=64:ompthreads=1:mem=64gb` | 1 | 64 | 00:40 | 64 GB | yes |
| E7 | `e7_layout_hc.pbs` | `auto` → batch_cpu | `8:ncpus=32:mpiprocs=32:ompthreads=1` | 8 | 256 | 00:25 | site hook 222 GB/chunk | yes |

No job requests more than 720 ncpus or 10 chunks. E2/E6-256/E7 are 256-core
jobs; E6-128 and E6-64 are the smaller fixed budgets required by E6. E5 uses
one 64-core node, matching the maximum practical OpenMP team.

## 3. Layout arithmetic (no oversubscription)

### E6-256, 4 nodes × 64 cores

| Layout | Ranks | Threads/rank | Ranks/node | Cores used/node | Ranks×threads |
|---|---:|---:|---:|---:|---:|
| 256×1 | 256 | 1 | 64 | 64 | 256 |
| 128×2 | 128 | 2 | 32 | 64 | 256 |
| 64×4 | 64 | 4 | 16 | 64 | 256 |
| 32×8 | 32 | 8 | 8 | 64 | 256 |
| 16×16 | 16 | 16 | 4 | 64 | 256 |
| 8×32 | 8 | 32 | 2 | 64 | 256 |
| 4×64 | 4 | 64 | 1 | 64 | 256 |

### E6-128, 2 nodes × 64 cores

| Layout | Ranks | Threads/rank | Ranks/node | Cores used/node | Ranks×threads |
|---|---:|---:|---:|---:|---:|
| 128×1 | 128 | 1 | 64 | 64 | 128 |
| 64×2 | 64 | 2 | 32 | 64 | 128 |
| 32×4 | 32 | 4 | 16 | 64 | 128 |
| 16×8 | 16 | 8 | 8 | 64 | 128 |
| 8×16 | 8 | 16 | 4 | 64 | 128 |
| 4×32 | 4 | 32 | 2 | 64 | 128 |

### E6-64, 1 node × 64 cores

| Layout | Ranks | Threads/rank | Ranks×threads |
|---|---:|---:|---:|
| 64×1 | 64 | 1 | 64 |
| 32×2 | 32 | 2 | 64 |
| 16×4 | 16 | 4 | 64 |
| 8×8 | 8 | 8 | 64 |
| 4×16 | 4 | 16 | 64 |
| 2×32 | 2 | 32 | 64 |
| 1×64 | 1 | 64 | 64 |

E2 uses 256 ranks × 1 thread on 4×64 = 64 ranks/node. E7 uses 256 ranks × 1
thread on 8×32 = 32 ranks/node.

## 4. Memory estimate

* 10M resampled workload at 1 rank: 10M × 54 × 8 B = **4.32 GB** for the rows,
  plus the 251 MB source matrix. E5 1-thread and E6 1×64 fit in 64 GB.
* 10M at 256 ranks: 39,063 rows/rank (33.7 MB) plus one 251 MB source copy per
  rank. On a 64-rank node this is 16.1 GB source + 2.2 GB rows ≈ 18.3 GB; the
  64 GB/444 GB requests are sufficient.
* Original 581,012 at 256 ranks: 2,270 rows/rank + 251 MB source per rank;
  64 ranks/node ≈ 16.1 GB source. Fits.

## 5. Walltime estimate and safety margin

Measured anchors:

| Experiment | Measured workload | Time |
|---|---|---:|
| E2 original 4×64 | 27 rows, 1–256 ranks | ≈ 3 min |
| E3 large 10M 4×64 | 27 rows, 1–256 ranks | ≈ 6 min (E2 part of 1552703) |
| E4 weak 50k/rank | 27 rows, 1–256 ranks | ≈ 3 min |
| E5 OpenMP | 42 rows (2 datasets × 7 threads × 3) | estimated 8–12 min |
| E6 per budget | 18–21 rows + 6–7 preflights | 8–15 min normally; ≤32 min if every layout hits its timeout |
| E7 | 10 rows + placement prints | estimated 5 min |
| E0 | build + 20 checks | estimated 5–8 min |

Requested walltimes are 25–45 min, i.e. at least 3× the expected runtime. The
timeout wrapper kills any individual hung layout at 90 s (preflight) or 180 s
(timed run), so an E6 job cannot consume its whole allocation on one hang.
Walltime is **not** increased to hide a launcher hang.

## 6. Submission gating

E2, E5, E6-64/128/256 and E7 are submitted with
`-W depend=afterok:<E0 job id>`. E0 must pass all of:

* forced build and checksum verification,
* serial/OpenMP/MPI/hybrid hash equality on 20k Covertype,
* fixed-total 200003 hash equality,
* weak identity 1×20000 vs 4×5000,
* two-host placement,
* at least one balanced multi-node hybrid mapping with `total_workers=8`.

E0 writes the first working balanced mapping (`ppr:R:node`, `node` or `slot`,
with `--bind-to none`) to `HYBRID_LAUNCH`; E6 jobs refuse to run without that
file. If E0 fails, no production job is released.

## 7. Scheduler state at the time of writing

The PBS server (`stdct-mgmt-02`, port 15001) became unresponsive during this
session: `qstat` timed out and two `qsub` processes hung on the login node. The
hung `qsub` processes were killed before they could create duplicate jobs. No
production job was submitted while the scheduler was unresponsive. Submission
will be retried only when `qstat -u harry_chen` returns normally; the exact job
IDs are recorded in `results/vanda_harry/SUBMISSION_MANIFEST.md` and
`docs/HPC_LOG.md`.
