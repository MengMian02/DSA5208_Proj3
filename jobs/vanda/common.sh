#!/usr/bin/env bash
# Shared preflight for the Vanda PBS job scripts. Sourced, not executed.

set -u

preflight() {
  if [ ! -f jobs/vanda/vanda.env ]; then
    echo "ERROR: jobs/vanda/vanda.env missing (cwd=$(pwd))" >&2
    exit 2
  fi
  . jobs/vanda/vanda.env
  . jobs/vanda/vanda_env.sh
  if [ ! -d "$WORKDIR" ]; then
    echo "ERROR: WORKDIR=$WORKDIR does not exist" >&2
    exit 2
  fi
  mkdir -p "$RESULTS_DIR" "$LOGS_DIR" cache
  echo "job=${PBS_JOBID:-local} host=$(hostname)"
  echo "workdir=$WORKDIR results=$RESULTS_DIR logs=$LOGS_DIR"
  echo "nodelist=$(sort -u "$PBS_NODEFILE" 2>/dev/null | tr '\n' ' ')"
  echo "ncpus_requested=${PBS_NP:-unset}"
  echo "mpicxx=$(command -v mpicxx || echo MISSING) $(mpicxx --version 2>/dev/null | head -1)"
  echo "python3=$(command -v python3 || echo MISSING) $(python3 -V 2>&1)"
}

# Vanda requires compilation on a compute node. Build only when a binary is
# missing, so concurrent jobs never delete each other's executables.
build_cpu() {
  missing=0
  for binary in kmeans_serial kmeans_openmp kmeans_mpi kmeans_hybrid; do
    [ -x "$WORKDIR/$binary" ] || missing=1
  done
  if [ "$missing" = "1" ]; then
    echo "== building on $(hostname) =="
    ( cd "$WORKDIR" && make cpu ) || { echo "ERROR: build failed" >&2; exit 3; }
  else
    echo "== binaries present, no build needed =="
  fi
  for binary in kmeans_serial kmeans_openmp kmeans_mpi kmeans_hybrid; do
    [ -x "$WORKDIR/$binary" ] || { echo "ERROR: $binary missing" >&2; exit 3; }
  done
  ls -l "$WORKDIR"/kmeans_serial "$WORKDIR"/kmeans_mpi "$WORKDIR"/kmeans_hybrid
}

job_results_path() {
  echo "$RESULTS_DIR/vanda_$1_${PBS_JOBID}.csv"
}
