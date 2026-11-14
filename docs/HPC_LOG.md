# HPC log

Append-only record of what was actually submitted and observed on NUS Atlas.
Nothing here is estimated: unobserved values stay blank.

## 2026-11-06 (night) - Atlas, first submission round

Environment probe (login atlas8-c01, CentOS 7.9):
* CPU: 2 x AMD EPYC 7643, 80 CPUs visible, SMT off.
* Queues: parallel (enabled, started, min ncpus 12, max ncpus 96, max mem
  540gb, walltime <= 720h, max_run 4 jobs/user, max_run_res.ncpus 96/user);
  serial (max ncpus 1, max mem 15gb). normal and benchmark are disabled.
* Compute nodes cnode-33-43-*: 96 CPUs, 359 GB RAM, 2 x 48 cores, 4 NUMA
  nodes, SMT off. Several nodes offline at probe time.
* Toolchain: /usr/bin/gcc is 4.8.5 (no C++17) and the default mpicxx is
  mvapich2-1.9. GCC 7.3.0 + Open MPI 4.1.1 exist under /app1; loading them with
  "module" still fails to link because the login environment's LIBRARY_PATH
  makes ld pick mvapich2's libmpich.so.10 (undefined ibverbs symbols). An
  explicit PATH/LD_LIBRARY_PATH with LIBRARY_PATH and CPATH unset links and
  runs correctly - see jobs/atlas/atlas_env.sh.
* Python: system python3 is 3.6.8; /app1 python3.7 is required (the runner
  needs 3.7+).

| Job | Purpose | Result |
|---|---|---|
| 1034398 | 12-core smoke gate | exit 0, 28 s on cnode-33-43-25; serial, OpenMP, MPI 2, hybrid 2x2, hybrid 8x1/4x2/2x4 and weak 1x20000 vs 4x5000 all matched hash 13331398156540273710 |
| 1034400 | E1 first attempt | exit 1 after 27/30 rows: a concurrently starting job ran "make clean" and deleted kmeans_serial |
| 1034405 | E4 multi-node attempt 1 | exit 1 in 20 s: same "make clean" collision, and the failure report died on missing shlex.join (Python 3.7) |
| 1034408 | E4 multi-node attempt 2 | exit 1: "Number of procs 48 exceeds PPR 24:node" - Open MPI saw one node |
| 1034410 | E4 multi-node attempt 3 | received both nodes, then orted could not connect back to mpirun (no route / no common interface). Blocked pending site guidance; no further attempts |
| 1034407 | E1 rerun | exit 0, 2 m 24 s on cnode-33-43-49, 30/30 rows, both grids complete |
| 1034401 | E2 strong scaling | submitted, queued at snapshot time |
| 1034403 | E3 weak scaling | submitted, queued at snapshot time |
| 1034404 | E4 hybrid layouts | submitted, queued at snapshot time |

Fixes applied during the round: jobs/common.sh no longer runs "make clean"
(builds only when a binary is missing); scripts/bencher.py no longer uses
shlex.join and its failure reporter cannot raise; jobs/atlas/multinode_hybrid.pbs
sets pipefail and passes the PBS nodefile as an explicit hostfile.

Open items:
* Queue wait for the three 96-core jobs is unknown and must not be estimated.
  They need a completely idle 96-core node, and the parallel queue was
  congested at submission time.
* Multi-node hybrid needs the site's interconnect/launcher answer before it can
  be retried; see jobs/atlas/MANIFEST.md.

## Outcome (same night)

* E2 (1034401), E3 (1034403) and E4 (1034404) all started between 03:50 and
  03:52 once a 96-core node drained, and each completed with Exit Status 0:
  24, 24 and 36 CSV rows respectively. No job exceeded five minutes of
  walltime despite the hour requested.
* E1 (1034407) had already completed: 30/30 rows, exit 0.
* E5 required no job; its per-phase timings come from the E2/E3/E4 rows.
* Analysis: the Atlas login node has no pandas or matplotlib in either
  python3.6 or the python3.7 module, so results were copied back and
  scripts/analyze.py was run locally (results/atlas/, figures/atlas/).
  Recorded deliverables: results/atlas/SUMMARY.md and eight figures.
* Open item unchanged: multi-node hybrid (1034405/1034408/1034410) remains
  blocked on cross-node orted connectivity; single-node hybrid covers all 12
  rank/thread layouts.


## 2026-11-12 - Vanda (PBS Pro) first submission round

Probe findings (login stdct-login-01, RHEL 9.4):
* Scheduler is **PBS Pro**, server stdct-mgmt-02. The earlier SLURM scripts for
  Vanda were wrong: Slurm binaries exist under /cm/shared/apps/slurm but there
  is no slurm.conf and the controller cannot be resolved, so those scripts were
  deleted. /opt/pbs is the live client.
* Execution queues are from_route_only, so jobs are submitted to route queues:
  auto_free -> cpu_serial / cpu_parallel / gpu (charge rate 0);
  auto -> batch_cpu / batch_gpu (charge rate 1).
* cpu_parallel caps a job at 160 ncpus, so 256 cores is impossible there: a
  256-core job sent to auto_free is refused by every destination. batch_cpu
  (ncpus <= 720, nodect <= 10) is the only route to 256 cores.
* Compute nodes: Intel Xeon Gold 6438N, 2 x 32 cores, SMT off, ~503 GB RAM.
* Compilation must happen on a compute node (login and compute architectures
  differ); system gcc 11.4.1, MPI from openmpi/gcc/64/4.1.8, python3 3.9.18
  without pandas/matplotlib so the analysis stays on the laptop.
* OpenMPI cannot load its PBS integration (libpbs.so.0 unavailable), so it does
  not learn the node list from PBS and a multi-node launch needs an explicit
  --hostfile "$PBS_NODEFILE". The job scripts pass it.

| Job | Purpose | Resources | State |
|---|---|---|---|
| 1546643 | 12-core correctness gate | cpu_parallel, 1 node | exit 0, PASSED (serial, OpenMP, MPI 8/12, hybrid, weak all matched hash 13331398156540273710) |
| 1546646 | E1 serial scaling | cpu_serial, 1 cpu | queued |
| 1546647 | E2 strong scaling to 256 | batch_cpu, 4 x 64 cpus | queued |
| 1546648 | E3 weak scaling to 256 ranks | batch_cpu, 4 x 64 cpus | queued |
| 1546649 | E4 hybrid at 256 cores | batch_cpu, 4 x 64 cpus | queued |
| 1546650 | E4 hybrid inside one 72/64-core node | cpu_parallel, 1 x 72 cpus | queued |

Unverified path: cross-node MPI. The smoke gate was single-node; 128- and
256-rank points depend on ssh/orted between compute nodes, which cannot be
tested without a multi-node allocation. If those ranks fail, the log and the
job output will show it and the single-node points still stand.


## 2026-11-13 - Vanda round 2: launch-policy fix and queue diagnosis

The three 256-core jobs submitted on 12 Nov (1547713/4/5) ran at 19:22 and
failed completely, producing no CSV. Root cause found with a dedicated 2-node
diagnostic (job 1547708) rather than guessed:

* PBS "select=N" means N CHUNKS, not N nodes. Without "-l place=scatter" both
  chunks landed on one host, so every ppr-based launch was rejected. All
  multi-node scripts now carry place=scatter.
* OpenMPI's ras_tm/plm_tm plugins cannot load without PBS's libpbs, so mpirun
  fell back to ssh+orted, which fails here ("orted: command not found"), and
  plain mpirun silently put every rank on one node. Exposing /opt/pbs/lib makes
  OpenMPI use PBS's TM API: ranks then split across nodes and the application
  hash matches the single-process run exactly. --hostfile is no longer used.

Queue diagnosis on 13 Nov (all four jobs queued, no holds):
* cpu_serial and cpu_parallel are bound to node pool cpu_free (24 nodes: 20 x 64
  cores, 3 x 72); batch_cpu to cpu_node (199 nodes, mostly 72 cores).
* E4-72 asked for 72 cores from cpu_free, i.e. it competed for 3 of 24 nodes,
  and reported "Insufficient amount of resource: node_pool". Rescaled to a
  64-core budget (fits 20 nodes) with layouts 64x1..1x64.
* The 256-core jobs reported "Insufficient amount of resource: ncpus" while 38
  nodes had >=64 free cores: all our jobs sit at Priority = 0, the scheduling
  cycle is 20 min, and batch_cpu had 480 jobs queued against 166 running.
  Walltime cut from 2 h to 40/45 min so backfill can place them.
* Self-imposed limit worth remembering: max_run_res.ncpus = 720 per user on
  batch_cpu, so only two of the three 256-core jobs can ever run at once.

Round 2 submissions (after cancelling 1546650/1547713/1547714/1547715):

| Job | Experiment | Queue / pool | Shape | Walltime |
|---|---|---|---|---|
| 1551056 | E2 strong -> 256 ranks | batch_cpu / cpu_node | 4 x 64 cpus | 00:40 |
| 1551057 | E3 weak -> 256 ranks | batch_cpu / cpu_node | 4 x 64 cpus | 00:45 |
| 1551058 | E4 hybrid at 256 cores | batch_cpu / cpu_node | 4 x 64 cpus | 00:40 |
| 1551059 | E4 hybrid in one node | cpu_parallel / cpu_free | 1 x 64 cpus | 01:30 |


## 2026-11-14 - Vanda round 3: 256-core scaling obtained, E4-256 failed

Scheduler outage: pbs_server on stdct-mgmt-02 was down for part of 13-14 Nov
(host pingable, ports 15001-15003 refused). It recovered; queued jobs then ran.

Completed and validated (all 27 rows = 9 rank counts x 3 reps, 30 fixed iters):
| Job | Experiment | Nodes actually used | Result |
|---|---|---|---|
| 1551056 | E2 strong 4x64 | 4 x GN-A40-05x | 129.73x speedup at 256 ranks, 51% efficiency, ONE label hash across all rank counts |
| 1551057 | E3 weak  4x64 | 4 nodes | weak efficiency 80.2% at 256 ranks |
| 1551069 | E2 strong 8x32 | 8 x CN-0xx | 6.59x at 256 ranks, 3% efficiency |
| 1551070 | E3 weak  8x32 | 8 nodes | weak efficiency 35.8% at 256 ranks |

The two shapes are NOT a clean 4-vs-8 node comparison: the 4x64 job landed on
GN-A40 nodes and the 8x32 job on CN nodes, so node family and interconnect are
confounded with node count. The 8x32 slowdown is entirely collective time:
at 128 ranks the Allreduce takes 0.399 s of a 0.426 s fit (94%), against a
small collective share at 64 ranks. Report it as a topology observation, not
as a scaling result.

E4 at 256 cores (1551058 4x64, 1551071 8x32) produced NOTHING: cput stayed at
00:00:00 for 28+ and 24+ minutes, zero repetitions, no CSV. Both select lines
implied mpiprocs x ompthreads far above ncpus (64x64 = 4096 threads on a
64-core chunk; 32x32 = 1024 on a 32-core chunk), unlike the E2/E3 jobs which
used ompthreads=1 and ran normally. Scripts fixed (ompthreads=1) but NOT
resubmitted, per instruction.

Walltime extension on a running job is impossible here: qalter returns
"Unauthorized Request". A longer walltime must therefore be chosen at
submission time.


## 2026-11-14 18:00 - harry_chen audit, repair and redesigned campaign

Repository state on arrival: `main` at f70e611, clean tree. The previous local
`--resampled-total` work was not in `main`; it was recovered from
`stash@{0}` and applied on branch `repair-2026-11-14` (backup branch
`backup-before-repair` at f70e611). The teammate CUDA files were preserved.

Previous harry_chen jobs re-examined from their own logs and CSV rows:

| Job | What actually happened |
|---|---|
| 1552672 | E2 original: 27/27 rows, 5-node heterogeneous 256 cores, one hash. E3 weak: 27/27 rows, 50k/rank, 55.4% weak efficiency. Exit 0. |
| 1552703 | E2 large 10M: 27/27 rows, 4x64 GN-A40, one hash, 133x/52% at 256. Then E4 hybrid produced no row and the job was killed at 1821 s. |
| 1552704 | E4 hybrid 128: no row, cput 00:00:00, killed at 1217 s. Launcher/placement hang, not a K-means failure. |

Root cause confirmed for the hybrid failures: the runner used
`--map-by ppr:R:node:pe=T --bind-to core`, and the existing hybrid rows are
internally inconsistent (256x1 fit 0.027/0.769/1.087 s). The repair changes the
production mapping to `ppr:R:node --bind-to none` with `OMP_PROC_BIND=false`,
adds a one-iteration preflight and a 180 s timeout per timed run, verifies
ranks/threads/total_workers, and splits E0/E2/E5/E6-64/E6-128/E6-256/E7 into
independent PBS jobs with `afterok` dependencies on E0.

At 18:05-18:12 the PBS server port 15001 stopped answering. `qstat -B` and
`qstat -u harry_chen` timed out, and two `qsub` processes for the hybrid
diagnostic hung on the login node. Both hung qsub processes were killed before
they could create duplicate jobs; no production job was submitted while PBS
was unavailable. Submission of the redesigned campaign is retried only after
`qstat` responds.

The E0 gate must pass before any production job; the expected result files and
row counts are listed in `docs/EXPERIMENTS.md` and
`results/vanda_harry/expected_results.csv`.
