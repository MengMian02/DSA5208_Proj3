# HPC submission kit

Status: **Atlas jobs have been submitted (see jobs/atlas/MANIFEST.md). Vanda is
unavailable and nothing here has been run on it.**

## Atlas (verified, PBS Pro)

Everything below was observed on the cluster on 2026-11-05/06, not assumed.

| Fact | Value |
|---|---|
| Login node | atlas8-c01, CentOS 7.9, 2 x AMD EPYC 7643, 80 CPUs, SMT off |
| Scheduler | PBS Pro, server venus01 |
| parallel queue | enabled; min ncpus 12, max ncpus 96, max mem 540gb, walltime <= 720 h |
| parallel per-user | max_run 4 jobs, max_run_res.ncpus 96 (so 96-core jobs run one at a time) |
| serial queue | enabled; max ncpus 1, max mem 15gb |
| disabled queues | normal (route), benchmark |
| compute nodes | cnode-33-43-*: 96 CPUs, 359 GB, 2 x 48 cores, 4 NUMA nodes, SMT off |
| compiler | GCC 7.3.0 at /app1/centos6.3/gnu/gcc-7.3.0 (system gcc 4.8.5 cannot do C++17) |
| MPI | Open MPI 4.1.1 at /app1/centos7/gnu/openmpi-4.1.1 |
| python | /app1/centos6.3/gnu/python3/3.7.0 (system python3 is 3.6.8, too old) |

### The linker trap (important)

"module load gcc-7.3.0 openmpi/4.1.1" gives the right compilers, but the
inherited LIBRARY_PATH then makes the linker pick
/app1/centos6.3/gnu/mvapich2-1.9/lib/libmpich.so.10 and fail with undefined
ibverbs symbols. jobs/atlas/atlas_env.sh therefore sets PATH and
LD_LIBRARY_PATH explicitly and unsets LIBRARY_PATH/CPATH; that configuration
compiles and runs (verified with a C++17 + MPI + OpenMP program under mpirun).

Note also that a shell function cannot be used in a pipeline: "module load X |
head" runs in a subshell and silently does nothing.

### Submitting

    ssh E0550588@atlas8.nus.edu.sg
    cd ~/dsa5208

    qsub jobs/atlas/smoke.pbs          # 12-core correctness gate (queue minimum)
    qsub jobs/atlas/serial.pbs         # E1, serial queue, 1 core
    qsub jobs/atlas/strong_scaling.pbs # E2, 96 ranks on one node
    qsub jobs/atlas/weak_scaling.pbs   # E3, 96 ranks, 50000 generated obs/rank
    qsub jobs/atlas/hybrid_layouts.pbs # E4, 12 rank x thread layouts at 96 cores
    qsub jobs/atlas/multinode_hybrid.pbs  # E4 multi-node (blocked, see manifest)

Each job writes its own CSV, named with its job id, so concurrent jobs can
never overwrite one another. Job stdout/stderr goes to the submission
directory (persistent home).

### Checking

    qstat -u $USER          # queued / running
    qstat -x -u $USER       # including finished
    ls -l results/
    tail -40 km-<name>.o<jobid>

## Why the E2/E3 sweep stops at 96 ranks

The parallel queue caps a job at 96 CPUs and 96 CPUs per user concurrently, and
each node has exactly 96 physical cores (SMT off). 96 ranks x 1 thread is
therefore the largest single-node configuration the queue allows. The
originally discussed 256-core run needs either a larger queue limit or working
cross-node MPI; cross-node launch currently fails in orted (see manifest).

## Vanda (SLURM, not used)

jobs/vanda/*.slurm were written earlier and are kept for when Vanda returns.
They have never been executed; per Vanda's policy they build on the compute
node (BUILD_ON_COMPUTE_NODE=1). Nothing in them has been validated against a
real Vanda scheduler.
