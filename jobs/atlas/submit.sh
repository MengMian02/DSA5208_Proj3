#!/usr/bin/env bash
# Submit an Atlas (PBS Pro) benchmark job. NOT executed automatically.
#
#   jobs/atlas/submit.sh serial
#   jobs/atlas/submit.sh strong  4 64      # 4 nodes x 64 ranks = 256 MPI ranks
#   jobs/atlas/submit.sh weak    4 64
#   jobs/atlas/submit.sh hybrid  1 72      # 72-core allocation, all layouts
#   jobs/atlas/submit.sh hybrid  2 32      # multi-node: 32 ranks x 2 threads
#
# The resource request is built here because PBS directives are not
# shell-expanded inside the script body.
set -eu

mode=$1
nodes=1
ppn=8
if [ $# -ge 2 ]; then nodes=$2; fi
if [ $# -ge 3 ]; then ppn=$3; fi

threads=1
if [ "$mode" = "hybrid" ]; then threads=$ppn; fi

script="jobs/atlas/$mode.pbs"
[ -f "$script" ] || { echo "unknown mode '$mode' (serial|strong|weak|hybrid)" >&2; exit 2; }
[ -f jobs/cluster.env ] || { echo "fill jobs/cluster.env first" >&2; exit 2; }

ranks=""
case "$mode" in
  strong) ranks="1,4,16,32,64,128,256" ;;
  weak)   ranks="1,4,16,32,64,128,256" ;;
esac

echo "submitting $mode: select=$nodes:ncpus=$ppn:mpiprocs=$ppn:ompthreads=$threads"
qsub -j oe \
     -l "select=$nodes:ncpus=$ppn:mpiprocs=$ppn:ompthreads=$threads" \
     -v RANKS="$ranks",REPS=3,LAYOUTS="72x1,36x2,18x4,9x8,4x18,1x72" \
     "$script"
echo "check with: qstat -u $USER"
