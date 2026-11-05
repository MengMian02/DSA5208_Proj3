#!/bin/bash
#SBATCH --job-name=kmeans-strong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#
# E2 strong scaling on Vanda (SLURM). The header above is a smoke-run floor;
# jobs/vanda/submit.sh passes the real --nodes/--ntasks-per-node.
# Vanda has no compiler toolchain on the login node, so the build runs here on
# the compute node (jobs/common.sh does it when BUILD_ON_COMPUTE_NODE=1 or when
# a binary is missing).
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
. jobs/common.sh
preflight
load_modules
build_if_needed

export OMP_NUM_THREADS=1
export MPI_LAUNCHER="$MPI_LAUNCHER"
export MPI_FLAGS=$(mpi_flags)

RANKS=$(printenv RANKS || echo "1,4,16,32,64,128,256")
REPS=$(printenv REPS || echo "$REPETITIONS")
OUT="$RESULTS_DIR/e2_strong_$SLURM_JOB_ID.csv"

echo "nodes=$SLURM_JOB_NUM_NODES tasks=$SLURM_NTASKS ranks=$RANKS out=$OUT"
python3 scripts/run_mpi.py --experiment strong --ranks "$RANKS" --repetitions "$REPS" \
  --max-iter 30 --k 7 --results "$OUT"
status=$?
echo "runner exit=$status"
exit $status
