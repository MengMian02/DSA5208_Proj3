#!/usr/bin/env bash
# Toolchain environment for NUS Atlas, verified on atlas8-c01 (2026-11-05).
#
# Facts observed on the cluster:
#   * OS: CentOS 7.9, PBS Pro, login node atlas8-c01 = 2 x AMD EPYC 7643,
#     80 CPUs visible, SMT off.
#   * Parallel-queue nodes (cnode-33-43-*): 96 CPUs, 359 GB RAM each.
#   * Default /usr/bin/gcc is 4.8.5 (no C++17) and the default mpicxx is an old
#     mvapich2-1.9: both must be replaced explicitly.
#   * "module load gcc-7.3.0 openmpi/4.1.1" raises gcc to 7.3.0 and mpirun to
#     Open MPI 4.1.1, but the stock login environment also exports a
#     LIBRARY_PATH / LD_LIBRARY_PATH mix that makes the linker pick up
#     mvapich2-1.9/lib/libmpich.so.10 and fail on undefined ibverbs symbols.
#     Setting the paths explicitly below and unsetting LIBRARY_PATH/CPATH links
#     and runs correctly (verified with a C++17 + MPI + OpenMP test program and
#     mpirun -np 2).
#
# Every PBS job script sources this file. It deliberately does not call
# "module", so a job cannot inherit those broken search paths.

export ATLAS_OPENMPI=/app1/centos7/gnu/openmpi-4.1.1
export ATLAS_GCC=/app1/centos6.3/gnu/gcc-7.3.0
export ATLAS_PYTHON=/app1/centos6.3/gnu/python3/3.7.0

export PATH=$ATLAS_OPENMPI/bin:$ATLAS_GCC/bin:$ATLAS_PYTHON/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=$ATLAS_OPENMPI/lib:$ATLAS_GCC/lib64:$ATLAS_GCC/lib

unset LIBRARY_PATH
unset CPATH
unset LD_PRELOAD

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Deterministic placement; a job may override MPI_FLAGS before calling preflight.
export MPI_LAUNCHER=${MPI_LAUNCHER:-mpirun}
export MPI_FLAGS=${MPI_FLAGS:---map-by core --bind-to core}
