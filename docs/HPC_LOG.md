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

