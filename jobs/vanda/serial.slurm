#!/bin/bash
#SBATCH --job-name=kmeans-serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#
# E1 serial scaling on a Vanda compute node (login nodes must not be used).
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
. jobs/common.sh
preflight
load_modules
build_if_needed

OUT="$RESULTS_DIR/e1_serial_$SLURM_JOB_ID.csv"
REPS=$(printenv REPS || echo "$REPETITIONS")
python3 scripts/run_e1.py --implementation serial --grid both --repetitions "$REPS" \
  --max-iter 30 --results "$OUT"
status=$?
echo "runner exit=$status"
exit $status
