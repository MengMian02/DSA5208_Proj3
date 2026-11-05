#!/usr/bin/env bash
# Collect the facts the job scripts need. Run on an ALLOCATED COMPUTE NODE:
#
#   qsub -I -l select=1:ncpus=8 -l walltime=00:10:00     # Atlas (PBS Pro)
#   salloc --nodes=1 --ntasks=1 --cpus-per-task=8        # Vanda (SLURM)
#   bash jobs/probe_cluster.sh | tee jobs/probe_$(hostname).txt
#
# Never run benchmarks on a login node; this probe only reads configuration.
set -u

echo "== host =="
hostname; uname -a; date -u

echo "== cpu topology =="
if command -v lscpu >/dev/null 2>&1; then lscpu; fi
if command -v numactl >/dev/null 2>&1; then numactl --hardware; fi
echo "nproc=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)"
getconf -a 2>/dev/null | grep -Ei 'LEVEL1_DCACHE|LEVEL2_CACHE|LEVEL3_CACHE|NPROCESSORS' | head -20

echo "== memory =="
free -h 2>/dev/null || vm_stat 2>/dev/null | head -5

echo "== scheduler =="
for tool in qsub qstat qdel sbatch squeue scancel sinfo; do
  command -v $tool >/dev/null 2>&1 && echo "found: $tool"
done
qstat -Q 2>/dev/null | head -20
sinfo -s 2>/dev/null | head -20

echo "== environment modules =="
if command -v module >/dev/null 2>&1; then
  module avail 2>&1 | head -60
else
  echo "no environment-modules command"
fi

echo "== mpi and compilers =="
for tool in mpirun mpiexec ibrun srun mpicxx mpicc gcc g++ clang python3; do
  if command -v $tool >/dev/null 2>&1; then
    echo "--- $tool: $(command -v $tool)"
    case $tool in
      mpirun|mpiexec|mpicxx) $tool --version 2>&1 | head -3 ;;
      gcc|g++|clang|python3) $tool --version 2>&1 | head -2 ;;
    esac
  fi
done

echo "== file systems =="
pwd; df -h . 2>/dev/null | head -5
echo "scratch hints:"; ls -d /scratch /work /tmp 2>/dev/null

echo "== interconnects =="
command -v ibstat >/dev/null 2>&1 && ibstat | head -20
command -v ibv_devinfo >/dev/null 2>&1 && ibv_devinfo | head -20
ip -o link show 2>/dev/null | head -10
