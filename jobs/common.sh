#!/usr/bin/env bash
# Shared preflight for every scheduler script in jobs/.
# Sourced, not executed: it defines preflight, load_modules, build_if_needed.
# Every scheduler body starts inside WORKDIR, so the env file is found there.

set -u

preflight() {
  env_file=jobs/cluster.env
  override=$(printenv JOBS_ENV_FILE || true)
  if [ -n "$override" ]; then env_file=$override; fi
  if [ ! -f "$env_file" ]; then
    echo "ERROR: $env_file is missing. Copy jobs/cluster.env.example and fill it from a probe run." >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  . "$env_file"
  if grep -q CHANGE_ME "$env_file"; then
    echo "ERROR: $env_file still contains CHANGE_ME values. Run jobs/probe_cluster.sh first." >&2
    grep -n CHANGE_ME "$env_file" >&2
    exit 2
  fi
  if [ ! -d "$WORKDIR" ]; then
    echo "ERROR: WORKDIR=$WORKDIR does not exist" >&2
    exit 2
  fi
  mkdir -p "$RESULTS_DIR"
  echo "host=$(hostname)"
  echo "workdir=$WORKDIR results=$RESULTS_DIR cores_per_node=$NODE_CORES"
}

load_modules() {
  if [ -n "$(printenv MODULE_COMPILER || true)" ]; then module load "$MODULE_COMPILER"; fi
  if [ -n "$(printenv MODULE_MPI || true)" ]; then module load "$MODULE_MPI"; fi
  if [ -n "$(printenv MODULE_PYTHON || true)" ]; then module load "$MODULE_PYTHON"; fi
  echo "mpicxx -> $(command -v mpicxx || echo MISSING)"
  echo "python3 -> $(command -v python3 || echo MISSING)"
}

build_if_needed() {
  needed=0
  if [ "$(printenv BUILD_ON_COMPUTE_NODE || true)" = "1" ]; then needed=1; fi
  if [ ! -x "$WORKDIR/kmeans_mpi" ]; then needed=1; fi
  if [ "$needed" = "1" ]; then
    echo "building on this node (Vanda forbids login-node compilation)"
    ( cd "$WORKDIR" && make cpu ) || { echo "ERROR: build failed" >&2; exit 3; }
  fi
  ls -l "$WORKDIR"/kmeans_serial "$WORKDIR"/kmeans_mpi "$WORKDIR"/kmeans_hybrid
}

# Launch flags that keep one rank per physical core with explicit binding.
# A site that rejects a flag says so on stderr; record what actually happened
# instead of assuming the binding took effect.
mpi_flags() {
  launcher_name=$(basename "$MPI_LAUNCHER")
  case "$launcher_name" in
    srun) echo "--cpu-bind=cores --distribution=block:block" ;;
    *)    echo "--bind-to core --map-by core" ;;
  esac
}
