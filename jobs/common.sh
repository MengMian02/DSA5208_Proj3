#!/usr/bin/env bash
# Shared preflight for the Atlas PBS job scripts. Sourced, not executed.
# The job bodies start in the repository root, so jobs/cluster.env and
# jobs/atlas/atlas_env.sh resolve relative to it.

set -u

preflight() {
  if [ ! -f jobs/cluster.env ]; then
    echo "ERROR: jobs/cluster.env is missing" >&2
    exit 2
  fi
  . jobs/cluster.env
  . jobs/atlas/atlas_env.sh
  if [ ! -d "$WORKDIR" ]; then
    echo "ERROR: WORKDIR=$WORKDIR does not exist" >&2
    exit 2
  fi
  mkdir -p "$RESULTS_DIR" "$LOGS_DIR" cache
  echo "job=${PBS_JOBID:-local} host=$(hostname)"
  echo "workdir=$WORKDIR results=$RESULTS_DIR logs=$LOGS_DIR"
  echo "nodes=$(sort -u "$PBS_NODEFILE" 2>/dev/null | wc -l) ncpus_requested=${PBS_NP:-unset}"
  echo "mpicxx=$(command -v mpicxx) $(mpicxx --version 2>/dev/null | head -1)"
  echo "python3=$(command -v python3) $(python3 -V 2>&1)"
}

# Build only when a binary is missing. NEVER run "make clean" here: several
# jobs share one working directory, and deleting binaries that another running
# job is executing fails that job mid-sweep (observed on job 1034400).
build_cpu() {
  missing=0
  for binary in kmeans_serial kmeans_openmp kmeans_mpi kmeans_hybrid; do
    [ -x "$WORKDIR/$binary" ] || missing=1
  done
  if [ "$missing" = "1" ]; then
    echo "== building missing binaries on $(hostname) =="
    ( cd "$WORKDIR" && make cpu ) || { echo "ERROR: build failed" >&2; exit 3; }
  else
    echo "== binaries present, no build needed =="
  fi
  for binary in kmeans_serial kmeans_openmp kmeans_mpi kmeans_hybrid; do
    [ -x "$WORKDIR/$binary" ] || { echo "ERROR: $binary missing" >&2; exit 3; }
  done
  ls -l "$WORKDIR"/kmeans_serial "$WORKDIR"/kmeans_openmp "$WORKDIR"/kmeans_mpi "$WORKDIR"/kmeans_hybrid
}

# One result file per job: two jobs can never append to the same CSV.
job_results_path() {
  echo "$RESULTS_DIR/atlas_$1_${PBS_JOBID}.csv"
}
