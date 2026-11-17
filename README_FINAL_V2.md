# DSA5208 Project 3 — final submission (v2)

This package accompanies `DSA5208_Proj3_Final_Report_v2.pdf`. The report is
based on the authoritative `Today_DSA_5208_Project_3.zip` revision and combines:

* the teammate GitHub baseline at `c450025`;
* Harry's personal HPC campaign on `repair-2026-11-14`;
* validated E2 1-rank recovery 1563234;
* validated E6-256 256×1 recovery 1563235;
* E6-128 original/backup jobs, which were still queued/held and are reported
  as pending only.

## Project structure

```
report_final_v2/        Editable final LaTeX source, 20-page PDF, figures, tables
results/vanda/          Teammate GitHub baseline results
results/vanda_harry/    Personal results, manifests, download audit, recovery CSV
hpc_download/           Personal PBS logs, build certificate, checksums
scripts/                Benchmark runners, including hybrid-auto mapping
jobs/                   PBS templates, recovery and E6-128 backup scripts
tests/                  Correctness, weak-scaling, Allreduce and Metal tests
gpu/                    CUDA and Metal implementations
kmeans_*.cpp, kmeans_common.hpp, Makefile
docs/                   Implementation, HPC log and result audits
```

## Required dataset

Download Covertype from
<https://archive.ics.uci.edu/dataset/31/covertype> and place it at
`data/covtype.data`. The raw dataset is not packaged.

## Build

```bash
make cpu        # serial, OpenMP, MPI, Hybrid
make gpu        # Metal on macOS
make test       # correctness suites; unavailable backends skip
```

## Run

```bash
./kmeans_serial data/covtype.data 7 30 42 --fixed-iterations
OMP_NUM_THREADS=4 ./kmeans_openmp data/covtype.data 7 30 42 --fixed-iterations
mpirun -np 4 ./kmeans_mpi data/covtype.data 7 30 42 --fixed-iterations
OMP_NUM_THREADS=2 mpirun -np 4 ./kmeans_hybrid data/covtype.data 7 30 42 --fixed-iterations
```

GPU binaries are `./kmeans_metal` on macOS and the CUDA target under `gpu/`.
CUDA kernel time is measured with events; upload, readback and CPU
accumulation are separate.

## PBS examples

`jobs/vanda/` contains templates and the recovery scripts. Set `#PBS -P` to the
submitter's allocation; multi-node Vanda launches require the site PBS TM
environment. Example:

```bash
qsub -P <allocation> jobs/vanda/e2_strong_hc.pbs
qsub -P <allocation> -W depend=afterok:<jobid> jobs/vanda/e6_128_backup_hc.pbs
```

Do not submit the E6-128 backup without first checking its smoke job and
resource shape (`jobs/vanda/e6_128_backup_smoke_hc.pbs`).

## Result regeneration

```bash
cd report_final_v2
latexmk -pdf main.tex
```

The authoritative report's `analysis/make_report_assets.py` regenerates the
baseline tables/figures. Personal results are in
`results/vanda_harry/`; `DOWNLOAD_AUDIT.md` records provenance and checksums.
`RESULT_PROVENANCE_FINAL.csv` at package root maps every series to its source.

## Known limitations

* E6-128 is absent: original 1557221 queued, backup smoke 1563236 queued,
  production 1563237 held. No 128-worker timing is claimed.
* E6-256 has six unpinned multi-thread layouts plus a separately allocated
  pinned 256×1 recovery (0.32014 s); binding/allocation differ.
* E2 recovery 1563234 is supplementary because it ran on a different node
  family from the primary 4×64 curve.
* Unpinned 256-core Hybrid rows 1554129/1554130 are invalid and excluded.
* Allreduce microbenchmark has no committed output.
* The OpenMP sweep covers 1–64 threads.
* Team names/student IDs are placeholders to be completed by the authors.

## Report QA

`DSA5208_Proj3_Final_QA.md` records compilation, provenance, ZIP extraction and
grading checks. `DSA5208_Proj3_Final_Grading_Checklist.md` maps the marking
scheme to exact report sections, tables and figures.
