#!/bin/bash
#SBATCH --job-name=kmeans-weak
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#
# E3 weak scaling on Vanda. 50000 generated observations per rank; the source
# Covertype matrix is cached as one binary file that every rank reads instead
# of a 250 MB broadcast at 256 ranks.
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
PER_RANK=$(printenv PER_RANK || echo 50000)
SOURCE_BIN=$(printenv SOURCE_BIN || echo "$WORKDIR/cache/covtype_source.bin")
OUT="$RESULTS_DIR/e3_weak_$SLURM_JOB_ID.csv"

mkdir -p "$(dirname "$SOURCE_BIN")"
echo "ranks=$RANKS per_rank=$PER_RANK source_bin=$SOURCE_BIN out=$OUT"
python3 scripts/run_mpi.py --experiment weak --ranks "$RANKS" --repetitions "$REPS" \
  --per-rank "$PER_RANK" --max-iter 30 --k 7 --source-bin "$SOURCE_BIN" --results "$OUT"
status=$?
echo "runner exit=$status"
exit $status
