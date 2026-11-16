# DSA5208 Project 3 — final submission package

This package combines the verified GitHub baseline `c450025` with the personal
Vanda campaign on branch `repair-2026-11-14` (local HEAD when packaged:
`3ef17dc` plus uncommitted recovery/report work). The final PDF is
`report_final/main.pdf`.

## Contents

```
report_final/            Final LaTeX source, 19-page PDF, figures, tables, analysis
results/vanda/           GitHub baseline results (unchanged)
results/vanda_harry/     Personal results, download audit and manifest
hpc_download/            Personal PBS logs, build certificate and checksums
scripts/                 CPU/GPU benchmark runners and analysis
jobs/                    PBS templates, recovery and E6-128 backup scripts
tests/                    correctness, weak-scaling, Allreduce and Metal tests
gpu/                     CUDA and Metal implementations
kmeans_*.cpp, kmeans_common.hpp, Makefile
docs/                    audits, implementation and HPC notes
GRADING_CHECKLIST_FINAL.md
RESULT_PROVENANCE_FINAL.csv
FINAL_REPORT_QA.md
RECOVERY_STATUS.md
E6_128_BACKUP_VALIDATION.md
README_FINAL_SUBMISSION.md
```

Raw `data/covtype.data` is not redistributed. Download it from
<https://archive.ics.uci.edu/dataset/31/covertype> into `data/covtype.data`.
The submission intentionally excludes Git metadata, compiled binaries and
temporary LaTeX files.

## Build and run

```bash
make cpu
make test
./kmeans_serial data/covtype.data 7 30 42 --fixed-iterations
OMP_NUM_THREADS=4 ./kmeans_openmp data/covtype.data 7 30 42 --fixed-iterations
mpirun -np 4 ./kmeans_mpi data/covtype.data 7 30 42 --fixed-iterations
OMP_NUM_THREADS=2 mpirun -np 4 ./kmeans_hybrid data/covtype.data 7 30 42 --fixed-iterations
```

PBS scripts are templates; change `#PBS -P personal-harry_chen` to the
submitter's allocation. Multi-node Vanda launches need the site PBS TM
environment.

## Report

```bash
cd report_final
latexmk -pdf main.tex
```

`report_final/analysis/personal_assets.py` regenerates the personal tables and
figures from `results/vanda_harry/`; the baseline report assets are copied from
the GitHub report. `RESULT_PROVENANCE_FINAL.csv` maps every series to its
source.

## Results status

* Complete: GitHub E1/E2/E3/E4/pinned/CUDA/Metal/E9; personal E1, E3 10M,
  E4, E5, E6-64, E7; E2 1-rank recovery 1563234.
* Partial: personal E2 1557218 (no 1-rank; recovery is standalone) and
  E6-256 1557222 (no 256×1).
* Queued/pending: original E6-128 1557221; E6-256 recovery 1563235; E6-128
  backup smoke 1563236 and production 1563237. No result is claimed for these.
* Invalid: unpinned 256-core Hybrid 1554129/1554130, retained only as
  diagnostic evidence.

Team names and student IDs on the title page are placeholders to be completed
by the authors.
