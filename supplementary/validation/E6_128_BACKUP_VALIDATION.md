# E6_128_BACKUP_VALIDATION

## Scope

Independent backup for the original E6-128 job **1557221**, which remains
queued. The backup does not modify or cancel the original and writes unique
outputs.

## Parameters (must match)

| Item | Value |
|---|---|
| Dataset | deterministic fixed-total 10,000,000 observations |
| K / D / seed | 7 / 54 / 42 |
| Iterations | 30 fixed |
| Precision | float64 CPU |
| Total workers | 128 |
| Nodes | 2 × 64 cores |
| Layouts | 128×1, 64×2, 32×4, 16×8, 8×16, 4×32 |
| Repetitions | 3 per layout |
| Expected rows | 18 |

## Resource declaration

Both backup jobs use:

```text
select=2:ncpus=64:mpiprocs=64:ompthreads=1:mem=100gb
place=scatter
```

The old incorrect `ncpus=64:mpiprocs=64:ompthreads=64` shape is not used.
`ompthreads=1` describes the PBS slot, while the runner explicitly sets
`OMP_NUM_THREADS=t` for each layout. Each layout is validated to use
`ranks × threads = 128`.

## Launcher policy

`--binding hybrid-auto` in `scripts/run_mpi.py`:

* `128×1`: `--map-by core --bind-to core` (one rank per core across 2×64).
* `64×2`: `--map-by ppr:32:node --bind-to none`.
* `32×4`: `--map-by ppr:16:node --bind-to none`.
* `16×8`: `--map-by ppr:8:node --bind-to none`.
* `8×16`: `--map-by ppr:4:node --bind-to none`.
* `4×32`: `--map-by ppr:2:node --bind-to none`.

## Jobs

| Stage | Job ID | State at finalization | Purpose |
|---|---|---|---|
| Smoke | **1563236** | Q | 2×64 shape, 200k/2 iterations, 4 layouts, host/affinity diagnostics |
| Production | **1563237** | H, `afterok:1563236` | 10M/30 iterations, 6 layouts × 3 reps |
| Original | 1557221 | Q | untouched |

## Pre-submission checks

| Check | Status |
|---|---|
| PBS connectivity and queue checked | PASS |
| Duplicate-job risk / unique names | PASS (`km-hc-e6-128s`, `km-hc-e6-128b`) |
| `select`, `ncpus`, `mpiprocs`, `ompthreads` consistent | PASS |
| Memory and walltime bounded | PASS |
| Two-node placement | PASS (`place=scatter`) |
| Per-layout rank/thread product validated by runner | PASS |
| 128×1 pinned mapping available | PASS (teammate-validated pattern) |
| Multi-thread affinity not forced to one core | PASS (`bind-to none`, full node cpuset) |
| Smoke dependency on production | PASS (`afterok`) |
| Smoke actually completed | **PENDING** |
| 18/18 production rows validated | **PENDING** |
| Label hash / SSE consistency | **PENDING** |
| CPU affinity evidence | **PENDING** (smoke will print) |
| Timing variability | **PENDING** |

## Validation rules when results arrive

1. 18/18 distinct rows, no duplicate `(ranks,threads,repetition)`.
2. `rows=10,000,000`, `K=7`, `D=54`, `iterations=30`.
3. All layouts use 128 workers and one hash (or tolerance-justified).
4. No failed/timed-out repetition.
5. Host/affinity log consistent with the requested rank/thread shape.
6. Do not merge with 1557221 unless hardware and binding are comparable.

**No result is claimed until the smoke passes and the production CSV is
downloaded and validated.**


## Last verified scheduler evidence

- 2026-11-16 13:55 SGT: original 1557221 queued (`node_pool`), backup smoke
  1563236 queued (`node_pool`), production 1563237 held by
  `afterok:1563236`.
- A live Vanda query on 2026-11-17 failed at hostname resolution
  (`Could not resolve hostname vanda.nus.edu.sg`). Current queue status is
  therefore unverified.
- `QUEUED` in the provenance record reflects the last observed scheduler state.
  It does not mean the jobs completed or failed.
