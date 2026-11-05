#!/usr/bin/env bash
# Submit a Vanda (SLURM) benchmark job. NOT executed automatically.
#
#   jobs/vanda/submit.sh serial
#   jobs/vanda/submit.sh strong 4 64      # 4 nodes x 64 tasks = 256 ranks
#   jobs/vanda/submit.sh weak   4 64
#   jobs/vanda/submit.sh hybrid 1 72
#
# The cluster partition and account are read from jobs/cluster.env; nothing is
# hard-coded here.
set -eu

mode=$1
nodes=1
ppn=8
if [ $# -ge 2 ]; then nodes=$2; fi
if [ $# -ge 3 ]; then ppn=$3; fi

script="jobs/vanda/$mode.slurm"
[ -f "$script" ] || { echo "unknown mode '$mode' (serial|strong|weak|hybrid)" >&2; exit 2; }
[ -f jobs/cluster.env ] || { echo "fill jobs/cluster.env first" >&2; exit 2; }

# shellcheck disable=SC1091
. jobs/cluster.env
if grep -q CHANGE_ME jobs/cluster.env; then
  echo "jobs/cluster.env still contains CHANGE_ME" >&2
  exit 2
fi

cpus=1
if [ "$mode" = "hybrid" ]; then cpus=$ppn; fi

ranks=""
case "$mode" in
  strong) ranks="1,4,16,32,64,128,256" ;;
  weak)   ranks="1,4,16,32,64,128,256" ;;
esac

echo "submitting $mode: nodes=$nodes ntasks-per-node=$ppn cpus-per-task=$cpus"
sbatch --nodes="$nodes" \
       --ntasks-per-node="$ppn" \
       --cpus-per-task="$cpus" \
       --partition="$QUEUE" \
       --account="$PROJECT_ACCOUNT" \
       --export=ALL,RANKS="$ranks",REPS=3,LAYOUTS="72x1,36x2,18x4,9x8,4x18,1x72",PER_RANK=50000,BUILD_ON_COMPUTE_NODE=1 \
       "$script"
echo "check with: squeue -u $USER ; scontrol show job <jobid>"
