#!/bin/bash
#SBATCH --job-name=kmeans-hybrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#
# E4 hybrid layouts at an equal core budget on Vanda. --cpus-per-task equals the
# OpenMP thread count of the layout, so 72 cores is 1 node x 72 tasks x 1
# thread, 9 tasks x 8 threads, or any other split of the same budget.
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
. jobs/common.sh
preflight
load_modules
build_if_needed

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export MPI_LAUNCHER="$MPI_LAUNCHER"
export MPI_FLAGS=$(mpi_flags)

LAYOUTS=$(printenv LAYOUTS || echo "4x1,2x2,1x4")
REPS=$(printenv REPS || echo "$REPETITIONS")
OUT="$RESULTS_DIR/e4_hybrid_$SLURM_JOB_ID.csv"

echo "layouts=$LAYOUTS reps=$REPS out=$OUT"
python3 scripts/run_mpi.py --experiment hybrid --layouts "$LAYOUTS" --repetitions "$REPS" \
  --max-iter 30 --k 7 --results "$OUT"
status=$?
echo "runner exit=$status"
exit $status
