# Atlas submission manifest

Cluster: NUS Atlas, PBS Pro (server venus01), login node atlas8-c01.
Account: E0550588. Working directory: /home/svu/e0550588/dsa5208
All jobs were submitted from that directory, so PBS job output files and all
CSV results land in persistent home storage (20 GB quota, shared NFS).

Snapshot time of this file: 2026-11-06 00:14 SGT.

## Job table

| Job ID | Name | Queue | Resources | Walltime | Script | Result file | Status |
|---|---|---|---|---|---|---|---|
| 1034398 | km-smoke | parallel | 1 node, 12 cpus, 16 gb | 00:40 | jobs/atlas/smoke.pbs | logs/smoke_test_1034398.venus01.log | COMPLETED, exit 0, all checks passed |
| 1034400 | km-e1-serial | serial | 1 cpu, 8 gb | 01:30 | jobs/atlas/serial.pbs | results/atlas_e1_serial_1034400.venus01.csv | FAILED, exit 1: 27 of 30 rows (superseded by 1034407) |
| 1034401 | km-e2-strong | parallel | 1 node, 96 cpus, 64 gb | 01:00 | jobs/atlas/strong_scaling.pbs | results/atlas_e2_strong_1034401.venus01.csv | COMPLETED 03:50, exit 0, 24/24 rows |
| 1034403 | km-e3-weak | parallel | 1 node, 96 cpus, 200 gb | 01:30 | jobs/atlas/weak_scaling.pbs | results/atlas_e3_weak_1034403.venus01.csv | COMPLETED 03:51, exit 0, 24/24 rows |
| 1034404 | km-e4-hybrid | parallel | 1 node, 96 cpus, 64 gb | 01:00 | jobs/atlas/hybrid_layouts.pbs | results/atlas_e4_hybrid_1034404.venus01.csv | COMPLETED 03:52, exit 0, 36/36 rows |
| 1034405 | km-e4-multinode | parallel | 2 nodes x 48 cpus, 100 gb each | 02:00 | jobs/atlas/multinode_hybrid.pbs | none | FAILED, exit 1: killed by a concurrent `make clean` (see below) |
| 1034407 | km-e1-serial | serial | 1 cpu, 8 gb | 01:30 | jobs/atlas/serial.pbs | results/atlas_e1_serial_1034407.venus01.csv | COMPLETED, exit 0, 30 of 30 rows |
| 1034408 | km-e4-multinode | parallel | 2 nodes x 48 cpus | 02:00 | jobs/atlas/multinode_hybrid.pbs | none | FAILED: Open MPI saw a single node (ppr conflict) |
| 1034410 | km-e4-multinode | parallel | 2 nodes x 48 cpus | 02:00 | jobs/atlas/multinode_hybrid.pbs | none | FAILED: orted cannot connect between nodes - BLOCKED (see below) |

Jobs 1034400, 1034405, 1034408 and 1034410 produced no usable data. The E1
rerun (1034407) is the authoritative E1 measurement. Nothing was cancelled.
No job was resubmitted more than twice, and nothing is resubmitted again
without your instruction.

## Root causes found and fixed

1. **Concurrent "make clean" destroyed a running job's binaries.**
   Every job body called build_cpu(), which ran "make clean". Job 1034405
   started at 00:03:16 while job 1034400 was mid-sweep; the clean deleted
   kmeans_serial and E1 died after 27 of 30 rows with FileNotFoundError.
   Fix: jobs/common.sh no longer cleans - it builds only when a binary is
   missing and then verifies all four executables. Verified by E1 rerun
   (1034407, exit 0, 30/30 rows).

2. **Python 3.7 on Atlas has no shlex.join (added in 3.8).**
   The failure-reporting path raised AttributeError, which converted a single
   failed run into an aborted sweep and hid the real stderr. Fix:
   scripts/bencher.py now uses a shlex.quote based helper and record_failure()
   can never raise. Both fixes were on disk before jobs 1034401/1034403/1034404
   started, and those jobs read these files at run time, so the queued jobs
   already benefit from them.

3. **Multi-node launch is not possible with the default Open MPI setup.**
   Attempt 2 passed the nodefile explicitly (--hostfile "$PBS_NODEFILE"), which
   fixed the mapping error, and the job did receive two nodes
   (cnode-33-43-38 + cnode-33-43-44, 48 cpus each). It then failed with
   "an inability to create a connection back to mpirun due to a lack of common
   network interfaces and/or no route found between them": orted cannot reach
   mpirun across nodes. This is a site network/launcher configuration matter,
   not an application bug. No further attempts were made. Candidate fixes for
   the user to authorise (they need the site's interconnect name, which the
   probe does not settle): --mca oob_tcp_if_include <iface>,
   --mca btl_tcp_if_include <iface>, or the launcher the cluster documentation
   prescribes for multi-node jobs. Single-node hybrid (job 1034404) is
   unaffected and covers all 12 rank/thread layouts.

## How to check progress

    ssh E0550588@atlas8.nus.edu.sg
    cd ~/dsa5208

    qstat -u $USER                    # queued / running
    qstat -x -u $USER                 # include finished jobs
    qstat -f <jobid> | egrep "job_state|comment|Exit_status"
    ls -l results/                    # one CSV per job, named by job id
    wc -l results/atlas_*.csv         # 1 header + expected rows
    tail -40 km-<name>.o<jobid>       # PBS stdout+stderr (job summary at the end)

Expected row counts when a job has finished successfully: E1 30, E2 24
(8 rank counts x 3 repetitions), E3 24, E4 36 (12 layouts x 3 repetitions).
A missing row count means the job has not finished or has failed; check the
runner_exit line in the job log.

## Resource requests, and why they are what they are

Observed limits (qstat -Qf / qmgr) - not assumed:
* parallel queue: resources_min.ncpus = 12, resources_max.ncpus = 96,
  resources_max.mem = 540gb, walltime up to 720 h, max_run = 4 jobs per user,
  max_run_res.ncpus = 96 per user.
* serial queue: resources_max.ncpus = 1, resources_max.mem = 15gb.
* parallel compute nodes (cnode-33-43-*): 96 CPUs, 359 GB RAM, 2 sockets x 48
  cores, 4 NUMA nodes, SMT off (lscpu on cnode-33-43-25).

Consequences:
* The smoke test requests 12 cores because that is the queue minimum.
* E2/E3 stop at 96 ranks: one node, one rank per physical core, which is also
  the hard per-job CPU cap of the parallel queue and the per-user running cap.
* Because max_run_res.ncpus is 96 per user, the three 96-core jobs run one at a
  time; they will execute sequentially overnight.
* E3 requests 200 GB because each of the 96 ranks holds its own 252 MB copy of
  the preprocessed source matrix (about 24 GB) plus its 21 MB work block.
* E1 runs in the serial queue, so it cannot consume a parallel-queue slot.

## Reproducibility statements

* Fixed iteration count: every E1-E4 run uses --fixed-iterations with
  MAX_ITER = 30, K = 7, seed 42.
* Identical initialization: k-means++ is run on rank 0 (MPI) or in the single
  process and broadcast, so all ranks and all rank counts start from the same
  centroids; E3 additionally always initialises from the original Covertype
  source, never from an expanded dataset.
* Every CSV row carries host, CPU model, core count, compiler string, flags,
  MPI version, dataset, K, D, seed, iterations, stopping mode, ranks, threads
  and repetition number.
* Binaries are compiled on the cluster with GCC 7.3.0 and Open MPI 4.1.1,
  -O3 -std=c++17 -DNDEBUG (see jobs/atlas/atlas_env.sh for the exact paths and
  the linker pitfall that makes the stock login environment unusable).

## Outcome (E1-E5 closed)

E2, E3 and E4 all started within the same minute (03:50-03:52) once a 96-core
node drained, and each finished in well under the requested hour: E2 24/24
rows, E3 24/24 rows, E4 36/36 rows, all with Exit Status 0. E1 finished earlier
(1034407, 30/30 rows, exit 0). E5 needs no job of its own; its phase timings
came from the E2/E3/E4 rows.

Headline numbers (medians of 3 repetitions, N=581012, K=7, I=30):
* E2 strong scaling: 4.415 s on 1 rank -> 0.0502 s on 96 ranks = 88.0x
  speedup, 91.7% parallel efficiency. Collective time never exceeds 0.013 s,
  rank imbalance stays below 0.9 ms.
* E3 weak scaling: fit time 0.380 -> 0.487 s while the workload grows from
  50 k to 4.8 M observations; efficiency 100/100/96.9/99.0/99.7/87.7/67.4/78.1%
  at 1/2/4/8/16/32/64/96 ranks. The dip at 64 ranks is a real measurement
  (collectives 0.189 s) and is not smoothed away.
* E4 hybrid at 96 cores: 96x1 0.0524 s, 48x2 0.0525 s, 12x8 0.0522 s,
  24x4 0.0526 s, but 1x96 0.0647 s and 6x16 0.0701 s. Pure MPI is as fast as
  the best hybrid, and putting all 96 threads in one rank is 23% slower
  because its local compute becomes 0.063 s against 0.050 s.

## Analysing the results

The Atlas login node has no pandas and no matplotlib (system python3 is 3.6.8
and the python3.7 module ships no site-packages), so plotting happens on the
local machine:

    scp -r E0550588@atlas8.nus.edu.sg:dsa5208/results/ results/atlas/
    python3 scripts/analyze.py --results results/atlas --figures figures/atlas \
        --experiments e1,e2,e3,e4,e5

That produced results/atlas/SUMMARY.md and the eight figures in figures/atlas/.
The scripts themselves are Python 3.6 compatible now, so on the cluster they
fail with a clear "needs pandas/matplotlib" message instead of a SyntaxError.

