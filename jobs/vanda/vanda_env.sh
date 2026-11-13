#!/usr/bin/env bash
# Vanda toolchain, verified by probe and by the km-diag2b launch test (2026-11-12).
#
# Observed cluster facts:
#   * PBS Pro, server stdct-mgmt-02. No usable Slurm (no slurm.conf, controller
#     unresolvable), so all job control is PBS.
#   * Compute nodes are HETEROGENEOUS: cn-* have 72 cores, in-* have 64.
#   * Queues: cpu_serial (1 core), cpu_parallel (2..160 ncpus, charge 0),
#     batch_cpu (up to 720 ncpus, charge 1). Execution queues are
#     from_route_only, so submit to route queues: auto_free (free) or auto
#     (charged). select=N means N CHUNKS, not N nodes - multi-node jobs must add
#     "-l place=scatter" or PBS may pack the chunks onto one host.
#   * Compilation must happen on a compute node (login and compute architectures
#     differ). System gcc 11.4.1; MPI from openmpi/gcc/64/4.1.8.
#
# Multi-node launch (this is the important part):
#   OpenMPI 4.1.8 here has ras_tm/plm_tm plugins, but they cannot load without
#   PBS's libpbs, so by default OpenMPI falls back to ssh+orted. On this cluster
#   that fallback fails ("orted: command not found") or silently places every
#   rank on one node. Exposing /opt/pbs/lib makes OpenMPI use PBS's TM API to
#   get the node list and launch, which was verified to spread ranks across
#   nodes with correct results. Do NOT pass --hostfile "$PBS_NODEFILE".

module load openmpi/gcc/64/4.1.8 2>/dev/null

# Only MULTI-NODE launches need PBS's TM integration. A single-node job that
# exposes libpbs wedged inside mpirun with zero CPU time (job 1551059), while a
# single-node job without it launches fine (smoke job 1546643). So gate it on the
# allocation actually spanning more than one host.
if [ -n "${PBS_NODEFILE:-}" ] && [ -d /opt/pbs/lib ]; then
  if [ "$(sort -u "$PBS_NODEFILE" | wc -l)" -gt 1 ]; then
    case ":$LD_LIBRARY_PATH:" in
      *":/opt/pbs/lib:"*) ;;
      *) export LD_LIBRARY_PATH=/opt/pbs/lib:$LD_LIBRARY_PATH ;;
    esac
  fi
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores

export MPI_LAUNCHER=${MPI_LAUNCHER:-mpirun}
export MPI_FLAGS=${MPI_FLAGS:---map-by core --bind-to core}
