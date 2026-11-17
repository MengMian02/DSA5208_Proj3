# RECOVERY_STATUS — final HPC recovery and E6-128 backup

Updated 2026-11-16 13:55 SGT. The original E6-128 job **1557221 remains
queued** and was not cancelled, modified or removed.

## Submitted jobs

| Purpose | Job ID | Request | Walltime | Dependency | State at finalization |
|---|---|---|---:|---|---|
| E2 1-rank × 1-thread recovery | **1563234** | 1 core, IN node | 00:15 | none | **F, exit 0**; 3/3 rows, median 3.429 s, hash `15912193932519828622` |
| E6-256 256×1 recovery | **1563235** | 4×64 cores | 00:30 | none | **F, exit 0**; 3/3 rows, median 0.32014 s, hash `14076111245358280480`; queued ~14 h 42 min |
| E6-128 backup smoke | **1563236** | 2×64 cores | 00:20 | none | **Q** at finalization |
| E6-128 backup production | **1563237** | 2×64 cores | 00:40 | `afterok:1563236` | **H** (held) |
| E6-128 original | 1557221 | 2×64 cores | 00:35 | — | **Q**, preserved |

No job was cancelled. No result is claimed for a queued/held job.

## Diagnosis

### E2 1557218
The PBS log `km-hc-e2strong.o1557218` records a 300 s timeout for
`mpirun --map-by core --bind-to core -np 1` inside a four-node allocation
(`CN-001`, `CN-003`, `CN-070`, `GN-A40-079`), before any application output.
Ranks 2--256 succeeded. The same binary and one-rank calculation had succeeded
in E0 on two IN nodes, so the failure is launcher/allocation-specific, not a
K-means computation failure. Recovery 1563234 isolated the singleton on one
IN-010 node with `--bind-to none`; it completed all three repetitions. Its
median 3.429 s is not merged into the 4×64 speedup curve because the original
campaign used a mixed CN/GN allocation and a different node family.

### E6-256 1557222
The log records the 256×1 preflight timing out at 90 s with
`mpirun --map-by ppr:64:node --bind-to none -np 256` on a mixed CN/GN
allocation. The six multi-thread layouts completed (18/21 rows). Recovery
1563235 used `--map-by core --bind-to core` on four CN nodes and completed
three repetitions (median 0.32014 s). It is a separate allocation, so the
seven measured configurations are reported with binding/placement caveats
rather than as a controlled same-allocation ranking.

### E6-128 1557221
Still queued for insufficient `node_pool`. It is **not** modified. The backup
is deliberately separate: 2×64 shape, `hybrid-auto` mapping (128×1 pinned;
multi-thread layouts `ppr:R:node` + `--bind-to none`), 10M fixed-total, K=7,
D=54, seed=42, I=30, three repetitions, six layouts = 18 rows.

## Code and script changes

* `scripts/run_mpi.py`: added `--binding hybrid-auto`. For one thread/rank it
  uses `--map-by core --bind-to core`; for multi-thread layouts it uses
  `--map-by ppr:R:node --bind-to none` with R = ranks/nodes. The mathematical
  implementation, preprocessing, seed, K, D and iteration count are untouched.
* `jobs/vanda/e2_rank1_recovery_hc.pbs`: single-node, one-core, `--binding none`.
* `jobs/vanda/e6_256_rank1_recovery_hc.pbs`: 4×64, `--binding hybrid-auto`.
* `jobs/vanda/e6_128_backup_smoke_hc.pbs`: 2×64, small 200k/2-iteration smoke,
  exact rank/thread shape, host/affinity diagnostics.
* `jobs/vanda/e6_128_backup_hc.pbs`: 2×64, 10M/30-iteration production, held
  behind the smoke by `afterok`.
* No existing PBS script or result CSV was overwritten.

## Safety checks performed

* Only `harry_chen` and `personal-harry_chen` were used.
* Queue checked before qsub; original 1557221 still present and untouched.
* `qsub` calls used 20 s bounds; PBS queries 10 s; no polling loop.
* Smoke and production use `select=2:ncpus=64:mpiprocs=64:ompthreads=1:mem=100gb`
  and `place=scatter`; no `ncpus=64:mpiprocs=64:ompthreads=64` declaration.
* Production is dependency-gated on the smoke.
* Observed queue waits from `ctime`/`stime`: E2 recovery 8 s; E6-256 recovery
  about 14 h 42 min; original E6-128 still queued for about 47 h; backup smoke
  still queued for about 22 h; production held by dependency.
* E2 recovery is complete; E6-256/E6-128 jobs are pending and are excluded from
  the report's measured curves.

## Remaining uncertainty

E6-256 recovery 1563235 completed and its three rows are validated and
included. The remaining queued/held E6-128 jobs (1557221, 1563236, 1563237)
have no results and are not part of any numerical claim.
