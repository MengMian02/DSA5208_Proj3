# Supplementary HPC experiments — DSA5208 Project 3

## 1. Purpose

This directory contains the personal Vanda experiments that extend the team's
main implementation: fixed-total 10M MPI strong scaling, Vanda OpenMP thread
scaling, Hybrid MPI+OpenMP layouts, node/rank placement diagnostics, and the
two targeted recoveries. It is self-contained and does not replace the main
source tree or its benchmark results.

## 2. Relationship to the main implementation

The core repository (`kmeans_*.cpp`, `kmeans_common.hpp`) remains the primary
implementation. The supplementary experiments need one additional option:
`--resampled-total N`, which materialises the same deterministic logical
N-row workload for every rank count. To avoid silently overwriting main, the
modified sources are kept in `src/` and compiled to this directory's root:

```bash
cd supplementary
make -f src/Makefile all
```

The differences from main are limited to fixed-total resampling, generated
workload bookkeeping and the associated tests. The serial, OpenMP, MPI and
Hybrid numerical algorithm, seed, K, D, precision and iteration count are
unchanged.

## 3. Software and hardware

* C++17, OpenMP, MPI (OpenMPI 4.1.8 on Vanda), Python 3.9+.
* Python packages for optional analysis: matplotlib, numpy (the summary script
  only needs the standard library).
* Vanda allocations used 64-core IN nodes, 4x64 GN-A40 nodes and 8x32 CN
  nodes. Exact Job IDs and walltimes are in `validation/PBS_EVIDENCE.md`.
* No GPU code is duplicated here; CUDA/Metal remain in the main repository.

## 4. Dataset

Download Covertype from
<https://archive.ics.uci.edu/dataset/31/covertype> and place it at:

```text
supplementary/data/covtype.data
```

The raw dataset is not committed. All scripts default to
`supplementary/data/covtype.data`; use `--data` to override.

## 5. Deterministic 10M resampling

`--resampled-total 10000000` does not create 10M new forest records. It builds a
fixed logical dataset by selecting rows from the preprocessed, class-ordered
Covertype source with a SplitMix64 hash of the global logical index. Every
rank count therefore sees the same N rows in the same order, with the same
initialisation and iteration count. `--weak-per-rank R` is the separate
weak-scaling mode where N grows as P*R. The two modes are mutually exclusive.

## 6. Build

```bash
cd supplementary
make -f src/Makefile all
```

Binaries are written to `supplementary/kmeans_serial`, `kmeans_openmp`,
`kmeans_mpi` and `kmeans_hybrid`; they are not copied into the main tree.
Use `make -f src/Makefile clean` to remove them.

For a Vanda allocation it is safer to build on the compute node:

```bash
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=1:mem=16gb
cd "$PBS_O_WORKDIR"
module load openmpi/gcc/64/4.1.8
make -f src/Makefile -B all
```

## 7. Run the experiments

From `supplementary/` after building:

```bash
# Serial E1, including 1M/5M/10M fixed-total points
python3 scripts/run_e1.py --implementation serial --grid both \
  --resampled-totals 1000000,5000000,10000000 \
  --results results/e1_serial.csv

# MPI strong scaling on 581k
python3 scripts/run_mpi.py --experiment strong --ranks 1,2,4,8,16,32,64,128,256 \
  --data data/covtype.data --k 7 --max-iter 30 --nodes 4 --binding core \
  --results results/e2_strong.csv

# Fixed-total 10M MPI strong scaling
python3 scripts/run_mpi.py --experiment strong --ranks 1,2,4,8,16,32,64,128,256 \
  --resampled-total 10000000 --data data/covtype.data \
  --nodes 4 --binding core --results results/e2_large.csv

# OpenMP 1..64 threads, 581k and 10M
python3 scripts/run_openmp.py --threads 1,2,4,8,16,32,64 \
  --data data/covtype.data --results results/e5_openmp_original.csv
python3 scripts/run_openmp.py --threads 1,2,4,8,16,32,64 \
  --resampled-total 10000000 --dataset-label covtype_resampled_10M \
  --results results/e5_openmp_10m.csv

# Hybrid layouts
python3 scripts/run_mpi.py --experiment hybrid --layouts 64x1,32x2,16x4,8x8,4x16,2x32,1x64 \
  --resampled-total 10000000 --nodes 1 --binding ppr-none \
  --preflight-rows 20000 --results results/e6_hybrid_64.csv
python3 scripts/run_mpi.py --experiment hybrid --layouts 256x1,128x2,64x4,32x8,16x16,8x32,4x64 \
  --resampled-total 10000000 --nodes 4 --binding ppr-none \
  --preflight-rows 20000 --results results/e6_hybrid_256.csv
```

`--binding hybrid-auto` is used by the recovery/backup scripts: one thread per
rank binds to cores; multi-thread layouts use `ppr:R:node` with `--bind-to
none`, because the `pe=T` forms hang under this Vanda/PBS TM launcher.

## 8. PBS submission

The PBS files are templates. Replace `#PBS -P YOUR_ALLOCATION` with the
submitter's project, and submit from `supplementary/` so `PBS_O_WORKDIR` and
`jobs/common.sh` resolve:

```bash
cd supplementary
qsub jobs/build_validate.pbs
qsub jobs/e1_serial.pbs
qsub jobs/e2_10m_strong.pbs
qsub jobs/e5_openmp.pbs
qsub jobs/e6_hybrid_64.pbs
qsub jobs/e6_hybrid_256.pbs
qsub jobs/e7_layout.pbs
qsub jobs/e2_rank1_recovery.pbs
qsub jobs/e6_256_rank1_recovery.pbs
# optional/pending 128-worker experiments:
qsub jobs/e6_128_backup_smoke.pbs
qsub -W depend=afterok:<smoke_id> jobs/e6_128_backup.pbs
```

`e6_128_original_pending.pbs` is retained only to document the original queued
job; it is not required for a fresh run. `build_validate.pbs` writes
`HYBRID_LAUNCH=ppr-none` and runs a short serial/OpenMP hash check; the Hybrid
scripts also default to `ppr-none` if the file is absent.

## 9. Expected outputs

The PBS helper writes `<jobid>_<suffix>.csv` under `results/`:

* `e1_serial`, `e2_strong`, `e2_large`;
* `e5_openmp_original`, `e5_openmp_10m`;
* `e6_hybrid_64`, `e6_hybrid_256`;
* `e7_*`, `e2_rank1_recovery`, `e6_256_rank1_recovery`;
* optional `e6_128_backup_*`.

Committed reference CSVs are listed in `results/INDEX.md`; their SHA-256
checksums are in `validation/checksums.sha256`.

## 10. Validation

```bash
python3 scripts/summarize_supplementary.py   # writes validation/summary.csv
python3 validation/run_resampled_tests.py    # needs binaries + data
sha256sum -c validation/checksums.sha256      # macOS: shasum -a 256 -c
```

`run_resampled_tests.py` checks rank-count-independent resampling across
serial/OpenMP/MPI/Hybrid and skips cleanly if a launcher is unavailable.
The committed evidence and hashes are described in
`validation/NUMERICAL_CHECKS.md`, `PBS_EVIDENCE.md` and `FAILURE_NOTES.md`.

## 11. Known limitations

* E6-128 is not complete: original 1557221 and backup smoke 1563236 are queued;
  production 1563237 is held. No 128-worker timing is included.
* The 256-worker set has six unpinned layouts and a separate pinned 256x1
  recovery (0.32014 s); binding/allocation are not a controlled single
  experiment.
* E2's 1-rank recovery (3.429 s) is supplementary; it ran on a different node
  family from the primary 4x64 curve.
* E7 placement logs for core/ppr32 lack the host printout; timing CSVs are
  complete.
* The OpenMP sweep covers 1--64 threads.
* The Allreduce microbenchmark has no committed output and is not included.
* Vanda queue waits, launcher timeouts and node-family differences are
  documented; they are part of the experimental limitations, not hidden.
