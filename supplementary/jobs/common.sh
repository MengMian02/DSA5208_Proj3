#!/usr/bin/env bash
# Shared helpers for supplementary Vanda jobs. Source this file; it deliberately
# contains no user-specific scratch path or account id.
set -u
SUPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SUPP_ROOT"
export WORKDIR="${WORKDIR:-$SUPP_ROOT}"
export RESULTS_DIR="${RESULTS_DIR:-$SUPP_ROOT/results}"
export LOGS_DIR="${LOGS_DIR:-$SUPP_ROOT/logs}"
export NODE_CORES="${NODE_CORES:-64}"
export MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
export MPI_FLAGS="${MPI_FLAGS:---map-by core --bind-to core}"
preflight() {
  command -v module >/dev/null 2>&1 && module load openmpi/gcc/64/4.1.8 2>/dev/null || true
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OMP_DYNAMIC=false
  export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
  export OMP_PLACES="${OMP_PLACES:-cores}"
  if [ -n "${PBS_NODEFILE:-}" ] && [ -d /opt/pbs/lib ]; then
    if [ "$(sort -u "$PBS_NODEFILE" | wc -l)" -gt 1 ]; then
      case ":$LD_LIBRARY_PATH:" in *":/opt/pbs/lib:"*) ;; *) export LD_LIBRARY_PATH=/opt/pbs/lib:${LD_LIBRARY_PATH:-} ;; esac
    fi
  fi
  mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
  echo "job=${PBS_JOBID:-local} host=$(hostname) date=$(date -Is)"
  echo "workdir=$WORKDIR nodes=$(sort -u "$PBS_NODEFILE" 2>/dev/null | tr '\n' ' ')"
}
require_binaries() {
  for binary in kmeans_serial kmeans_openmp kmeans_mpi kmeans_hybrid; do
    [ -x "$WORKDIR/$binary" ] || { echo "ERROR: $binary missing; build with 'make -f src/Makefile all'" >&2; exit 3; }
  done
}
job_results_path() { echo "$RESULTS_DIR/${PBS_JOBID:-local}_$1.csv"; }
