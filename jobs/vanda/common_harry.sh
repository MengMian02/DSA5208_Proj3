#!/usr/bin/env bash
# Shared preflight for harry_chen's Vanda jobs. Sourced, not executed.
# Deliberately separate from jobs/vanda/common.sh, which points at the
# teammate's /scratch/e0550588 workspace.

set -u

preflight() {
  . jobs/vanda/harry.env
  module load openmpi/gcc/64/4.1.8 2>/dev/null
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
  export OMP_DYNAMIC=false
  export OMP_PROC_BIND=close
  export OMP_PLACES=cores
  export MPI_LAUNCHER=${MPI_LAUNCHER:-mpirun}
  export MPI_FLAGS=${MPI_FLAGS:---map-by core --bind-to core}

  # Open MPI's PBS TM plugin needs libpbs. Only expose it for multi-node
  # launches: on a single node the TM path has been observed to wedge.
  if [ -n "${PBS_NODEFILE:-}" ] && [ -d /opt/pbs/lib ]; then
    if [ "$(sort -u "$PBS_NODEFILE" | wc -l)" -gt 1 ]; then
      case ":$LD_LIBRARY_PATH:" in
        *":/opt/pbs/lib:"*) ;;
        *) export LD_LIBRARY_PATH=/opt/pbs/lib:${LD_LIBRARY_PATH:-} ;;
      esac
    fi
  fi

  mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
  echo "job=${PBS_JOBID:-local} host=$(hostname) date=$(date -Is)"
  echo "workdir=$WORKDIR results=$RESULTS_DIR logs=$LOGS_DIR"
  echo "nodes=$(sort -u "$PBS_NODEFILE" 2>/dev/null | tr '\n' ' ')"
  echo "ncpus_requested=${PBS_NP:-unset}"
  echo "mpicxx=$(command -v mpicxx || echo MISSING) $(mpicxx --version 2>/dev/null | head -1)"
  echo "python3=$(command -v python3 || echo MISSING) $(python3 -V 2>&1)"
}

build_cpu_force() {
  echo "== forced rebuild on $(hostname) =="
  ( cd "$WORKDIR" && make -B cpu ) || { echo "ERROR: build failed" >&2; exit 3; }
  for binary in kmeans_serial kmeans_openmp kmeans_mpi kmeans_hybrid; do
    [ -x "$WORKDIR/$binary" ] || { echo "ERROR: $binary missing" >&2; exit 3; }
  done
  ( cd "$WORKDIR" && md5sum kmeans_serial kmeans_openmp kmeans_mpi kmeans_hybrid       > BUILD_INFO.md5 )
  {
    echo "build_time=$(date -Is)"
    echo "build_host=$(hostname)"
    echo "source_version=$(tr '\n' ';' < "$WORKDIR/SOURCE_VERSION" 2>/dev/null || echo MISSING)"
    echo "gcc=$(gcc --version 2>/dev/null | head -1)"
    echo "mpicxx=$(mpicxx --version 2>/dev/null | head -1)"
    echo "make_flags=$(grep -E '^BASE_FLAGS' "$WORKDIR/Makefile" 2>/dev/null | head -1)"
    echo "binaries:"
    cat "$WORKDIR/BUILD_INFO.md5"
  } > "$WORKDIR/BUILD_INFO.txt"
  cat "$WORKDIR/BUILD_INFO.txt"
}

require_binaries() {
  for binary in kmeans_serial kmeans_openmp kmeans_mpi kmeans_hybrid; do
    [ -x "$WORKDIR/$binary" ] || {
      echo "ERROR: $binary missing; run the E0 build gate first" >&2; exit 3; }
  done
  if [ ! -f "$WORKDIR/BUILD_INFO.md5" ]; then
    echo "ERROR: BUILD_INFO.md5 missing; run the E0 build gate first" >&2; exit 3
  fi
  ( cd "$WORKDIR" && md5sum -c --quiet BUILD_INFO.md5 ) || {
    echo "ERROR: binaries do not match BUILD_INFO.md5; rebuild in E0" >&2; exit 3; }
}

job_results_path() {
  echo "$RESULTS_DIR/${PBS_JOBID}_$1.csv"
}
