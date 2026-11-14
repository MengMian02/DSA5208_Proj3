# CODE_AUDIT — DSA5208 Project 3

Date: 2026-11-14 (Asia/Singapore)
Repository: `/Users/hc/Desktop/NUS DSML/DSA 5208/DSA5208_Proj3`
Branch: `repair-2026-11-14` (created from `main` at `f70e611`)
Backup branch: `backup-before-repair` (points at `f70e611`)
HPC account: `harry_chen` only. `/scratch/e0550588` was never accessed.

This audit separates **confirmed defects** (reproduced from source, logs or
CSVs) from **hypotheses** that still need an experiment.

## 1. Repository state found

The working tree was clean at `f70e611` ("Added GPU result"). The previous
local HPC work was not in `main`; it was preserved in
`stash@{0}` ("My local HPC changes before teammate pull"):

* `kmeans_common.hpp`, `kmeans_mpi.cpp`, `kmeans_hybrid.cpp`,
  `kmeans_openmp.cpp`, `kmeans_serial.cpp`
* `scripts/run_mpi.py`, `scripts/run_e1.py`, `scripts/analyze.py`
* `README.md`, `docs/HPC_LOG.md`

The stash was applied selectively on the repair branch; the new CUDA files
(`gpu/cuda/kmeans_cuda.cu`, `scripts/run_cuda.py`, `jobs/vanda/gpu_*.pbs`,
`results/vanda/e7_cuda_1554475.csv`) were left untouched. No historical result
was deleted.

## 2. Existing functionality

| Component | State |
|---|---|
| Serial K-means | present, fixed-total path restored |
| OpenMP K-means | present, thread-local accumulation, fixed-total restored |
| MPI K-means | present, strong/weak modes, generated-workload path restored |
| Hybrid MPI+OpenMP | present, `MPI_THREAD_FUNNELED`, thread-local sums |
| Apple Metal | present, float32 GPU assignment, CPU double reference |
| NVIDIA CUDA | present (`gpu/cuda/kmeans_cuda.cu`), A40 results 30 rows |
| Benchmark runners | `run_e1.py`, `run_mpi.py`, `run_cuda.py`, new `run_openmp.py` |
| Tests | `tests/run_tests.py`, `run_weak_scaling_tests.py`, `run_metal_tests.py` |
| PBS kit | teammate Atlas/Vanda scripts plus `harry_chen` scripts |
| Existing results | Atlas, Vanda teammate, own Vanda `results/vanda_harry` |

## 3. Confirmed defects

### C1. Fixed-total resampling was missing from `main` (confirmed)

Evidence:

```
$ grep -n resampled_total kmeans_common.hpp kmeans_mpi.cpp
(no output before the stash was restored)
```

`main` supported only `--rows` (systematic subset ≤ source size) and
`--weak-per-rank` (N grows with rank count). A fixed N=10M strong-scaling run
was impossible. The stash restores:

* `Options::resampled_total` and `--resampled-total N`,
* `generated_workload()` / `generated_total()`,
* generated-mode branches in MPI and hybrid,
* serial/OpenMP materialisation of the same logical dataset,
* `run_mpi.py --resampled-total` and `run_e1.py --resampled-totals`.

Fix: restored and extended; serial/OpenMP now reject `--weak-per-rank` instead
of silently ignoring it.

### C2. Hybrid `pe=` launch hangs under Vanda PBS TM (confirmed)

Evidence:

* job **1552704** (2×64, 128-core hybrid) printed its result path and then
  produced no repetition for 1217 s; PBS killed it at the 1200 s walltime.
  `cput` was zero, so the application never started.
* job **1552703** completed E2-large (27 rows in `..._e2_large.csv`) and was
  killed inside the E4 hybrid section after 1821 s.
* the previously "successful" hybrid CSVs are internally inconsistent:
  job 1554129, layout 256×1: `fit_s` = 0.0273, 0.7690, 1.0875 s in three
  repetitions; layout 8×32: 3.42–3.55 s.
* `scripts/run_mpi.py` used
  `--map-by ppr:R:node:pe=T --bind-to core` for hybrid. The teammate
  diagnostics `diag_mapping.pbs` / `diag_pe.pbs` found that every `pe=` form
  hangs or is rejected under this launcher.

Fix: E0 tests `ppr:R:node`, `node`, `slot`, `none` and `core` mappings without
`pe=`, then writes the first balanced working mapping to `HYBRID_LAUNCH`.
Production hybrid reads that file, uses `--bind-to none`,
`OMP_PROC_BIND=false`, an exclusive PBS allocation, and a preflight/timeout per
layout. Every layout now gets a one-iteration preflight with a hard
timeout; a failing layout is logged and skipped.

### C3. PBS allocation metadata implied impossible thread counts (confirmed)

Historical `hybrid_256.pbs` used
`ncpus=64:mpiprocs=64:ompthreads=64` (4096 threads on a 64-core chunk) and
`hybrid_256_8x32.pbs` used `ncpus=32:mpiprocs=32:ompthreads=32` (1024 threads
on a 32-core chunk). Both E4 jobs produced no rows (HPC_LOG 2026-11-14).

Fix: all new scripts request `ompthreads=1` as allocation metadata and set
`OMP_NUM_THREADS` per run inside the runner. The runner now checks the
reported `ranks`, `threads` and `total_workers` against the requested layout.

### C4. Teammate-specific account/project paths (confirmed)

Current `jobs/vanda/*.pbs` and `jobs/vanda/common.sh` still contain:

* `#PBS -P personal-e0550588`
* `WORKDIR=/scratch/e0550588/dsa5208`
* `RESULTS_DIR=$WORKDIR/results`

Submitting them as `harry_chen` would fail or write into the teammate's
workspace. Fix: a separate `jobs/vanda/harry.env`, `jobs/vanda/common_harry.sh`
and `*_hc.pbs` kit use only `personal-harry_chen` and
`/scratch/harry_chen/DSA5208_Proj3`.

### C5. Stale-binary risk (confirmed)

`jobs/vanda/common.sh` builds only when a binary is missing:

```
for binary in ...; do [ -x "$WORKDIR/$binary" ] || missing=1; done
```

A source edit can therefore leave an old executable in place. `common_harry.sh`
now performs a forced `make -B cpu`, writes `BUILD_INFO.md5` and
`BUILD_INFO.txt` (compiler, MPI, flags, build host/time, source version), and
every production job runs `md5sum -c --quiet BUILD_INFO.md5` before using a
binary.

### C6. No per-run timeout (confirmed)

`scripts/run_mpi.py` called `bencher.execute` with the default 3600 s timeout
and one monolithic job. A single hung hybrid layout consumed the complete
30-minute allocation (1552703) or 20-minute allocation (1552704).

Fix: `--timeout` per timed run, `--preflight-timeout` per layout, an OS
`timeout -k 10` wrapper, failure logging, and independent PBS jobs per
experiment/core budget.

### C7. Worker count was trusted, not verified (confirmed)

The runner recorded `metrics["total_workers"]` without comparing it with
`ranks × threads`. A layout that silently launched fewer ranks would still be
written as a valid row.

Fix: `check_worker_count()` now rejects any mismatch.

### C8. Integer-overflow gaps (confirmed by inspection; guard added)

`generated_total()` multiplied `weak_per_rank * ranks` without an overflow
check, and `materialise_indices()` sized `count * D` without checking
`size_t`. Both are now guarded.

### C9. CUDA runner leaves two schema columns blank (confirmed, minor)

`GPU_SCHEMA` has `gpu_kernel_total_s` and `gpu_host_total_s`, but
`run_cuda.py` does not fill them. The existing CUDA rows are still usable for
kernel/fit/transfer analysis; this is a reporting gap, not a correctness bug.

### C10. Serial/OpenMP accepted an MPI-only option (confirmed)

`--weak-per-rank` was parsed by serial/OpenMP and silently ignored, producing a
row labelled weak=0 for a workload the user did not request. Fixed: both now
exit with an explicit error.

## 4. Performance anomalies that are NOT yet explained

### A1. 4×64 vs 8×32 strong scaling difference

From the committed CSVs (median of 3):

| Layout | Rank 1 fit | 256-rank fit | Speedup | Efficiency | 128/256 collective |
|---|---:|---:|---:|---:|---:|
| 4×64 (GN-A40-051) | 2.887 s | 0.0223 s | 129.7× | 50.7% | 0.012 / 0.010 s |
| 8×32 (CN-006) | 2.883 s | 0.4374 s | 6.59× | 2.6% | 0.399 / 0.423 s |

Both report the same CPU model string (Xeon Platinum 8452Y) and the same
label hash `15912193932519828622`. The 8×32 loss is almost entirely the
collective: at 128 ranks the Allreduce takes 0.399 s of a 0.426 s fit. This is
**not** evidence that fewer nodes are inherently faster. Candidate causes:

1. GN-A40 vs CN node family / interconnect path (hardware confound).
2. A bad rank placement or an oversubscribed/contended node in the 8-node
   allocation.
3. OpenMPI transport selection on the CN fabric.

E7 will rerun 128/256 ranks on one 8×32 CN allocation with three explicit
mappings and print per-host rank counts. Until that data exists, any statement
that "8 nodes are slow" is a hypothesis.

### A2. First-repetition warm-up

In the 4×64 E2 CSV the 1-rank row is 3.607 s for rep 1 and 2.880/2.884 s for
reps 2–3. All reported tables use the median, so the warm-up does not move the
median, but the raw CSV must not be read as three independent steady-state
measurements.

### A3. Hybrid 256×1 variance

0.0273 / 0.7690 / 1.0875 s for the same layout is a factor of 40. A correct
hybrid run should not vary that much; this is another symptom of C2, not
K-means noise.

## 5. Resource configuration findings

* Vanda PBS Pro: `batch_cpu` max 720 ncpus, max 10 chunks, max 6 running jobs,
  720 ncpus/user; `cpu_parallel` max 160 ncpus; `cpu_serial` max 1.
* `select=N` means N chunks, not N nodes; `place=scatter` is required for
  multi-node allocations.
* OpenMPI 4.1.8 + PBS TM needs `/opt/pbs/lib` for multi-node launches; single
  node launches should not expose it (observed to wedge in 1551059).
* The batch_cpu memory hook reserves memory proportional to ncpus; a
  4×64 job is recorded as `mem=444GB` per chunk, so it needs four nearly empty
  72-core nodes. This is a schedulability constraint, not a bug in the code.
* `ompthreads` in `select` is allocation metadata; it does not set
  `OMP_NUM_THREADS`. The runner must set it and verify it.

## 6. Required changes (implemented on the repair branch)

| Change | File(s) | Status |
|---|---|---|
| Restore fixed-total resampling | `kmeans_common.hpp`, four CPU sources, `run_mpi.py`, `run_e1.py` | done |
| Overflow guards | `kmeans_common.hpp` | done |
| Serial/OpenMP reject weak mode | `kmeans_serial.cpp`, `kmeans_openmp.cpp` | done |
| Hybrid `ppr-none` mapping + worker check | `scripts/run_mpi.py` | done |
| Per-run and preflight timeouts | `scripts/run_mpi.py` | done |
| OpenMP scaling runner | `scripts/run_openmp.py` | new |
| Build stamp and checksum verification | `jobs/vanda/common_harry.sh` | done |
| Independent E0/E2/E5/E6/E7 PBS scripts | `jobs/vanda/*_hc.pbs` | new |
| Real-data analysis | `scripts/analyze_covertype.py` | new |
| Stale teammate project/account isolated | own `harry.env` / `common_harry.sh` | done |

## 7. Tests required before production

1. Build on a Vanda compute node with `make -B cpu`; record checksums.
2. Serial/OpenMP/MPI 1,2,4/hybrid 4×2 hash equality on 20k Covertype.
3. Same four implementations on `--resampled-total 200003`.
4. Weak identity 1×20000 vs 4×5000.
5. Two-node rank placement (exactly two hosts).
6. Hybrid multi-node launch variants with timeouts; at least one must produce
   `total_workers=8` and the reference hash.
7. Worker-count validation for each hybrid layout.
8. Production jobs only after E0 passes.

## 8. Evidence index

* `results/vanda_harry/1552672..._e2_strong_256_hetero.csv` (27 rows)
* `results/vanda_harry/1552672..._e3_weak_256_hetero.csv` (27 rows)
* `results/vanda_harry/1552703..._e2_large.csv` (27 rows)
* `km-hc-10m.o1552703`, `km-hc-hyb128.o1552704` (Vanda logs)
* `results/vanda/vanda_e4_hybrid256_1554129...csv` (21 rows, invalid variance)
* `results/vanda/vanda_e2_strong_1551056...csv`, `..._8x32_1551069...csv`
* `results/vanda/e7_cuda_1554475.csv` (30 rows, agreement 1.0 or 0.9999875)
* `docs/HPC_LOG.md` (PBS findings, 2026-11-13/14)

## 9. Post-repair compile checks and scheduler state

At 18:21 SGT the repaired sources were syntax-checked on the Vanda login node
with the compute toolchain (`g++` 11.4.1, OpenMPI 4.1.8):

```
g++     -fsyntax-only kmeans_serial.cpp              OK
g++     -fopenmp -fsyntax-only kmeans_openmp.cpp     OK
mpicxx  -fsyntax-only kmeans_mpi.cpp                 OK
mpicxx  -fopenmp -fsyntax-only kmeans_hybrid.cpp     OK
python3 -m py_compile scripts/*.py                   OK
```

No binaries were produced on the login node. At 18:05-18:40 the PBS server
`stdct-mgmt-02:15001` was closed and `qstat` timed out; the two hung `qsub`
processes were stopped. The E0/E2/E5/E6/E7 kit is prepared and uploaded, but
no production job has been submitted while the scheduler is unreachable.
