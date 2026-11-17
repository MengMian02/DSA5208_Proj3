# SUPPLEMENTARY_INTEGRATION_AUDIT

## Base and branch

* Base: `origin/main` = `c450025baf9e3cdc173646d2ccb6063e731a189f`.
* Integration branch: `feature/supplementary-hpc-experiments`.
* Personal branch preserved: `repair-2026-11-14` at `ab64b19` (remote).
* No merge of personal history and no change to `main`.

## Files added

* `supplementary/src/`: modified `kmeans_common.hpp`, `kmeans_serial.cpp`,
  `kmeans_openmp.cpp`, `kmeans_mpi.cpp`, `kmeans_hybrid.cpp`, and
  `Makefile` for a standalone build into `supplementary/`.
* `supplementary/scripts/`: `bencher.py`, `run_e1.py`, `run_mpi.py`,
  `run_openmp.py`, `summarize_supplementary.py`.
* `supplementary/jobs/`: `common.sh`, build/validation, E1 serial, E2 581k and
  10M, E5 OpenMP, E6 Hybrid 64/256, E7 placement, E2/E6 recovery and optional
  E6-128 backup/pending scripts.
* `supplementary/results/`: 13 verified CSVs (E1, 10M E3, weak E4, E2 partial
  + recovery, E5 original/10M, E6-64, E6-256 partial + recovery, E7 core/
  ppr32/ppr16) and `INDEX.md`.
* `supplementary/validation/`: build certificate, source version, selected PBS
  logs, checksums, numerical checks, PBS evidence, failure notes, recovery/
  backup docs, report provenance and traceability.
* `supplementary/README.md` and `supplementary/data/README.md`.
* Root README section `Supplementary HPC Experiments`.

## Files modified in main tree

* `README.md`: one appended section pointing to `supplementary/README.md`.
  No existing instructions or results were changed.

## Duplicates excluded

* Main `kmeans_*.cpp`/`kmeans_common.hpp` are not copied to the repository
  root; the modified copies are isolated under `supplementary/src/`.
* Main GPU source (`gpu/cuda`, `gpu/metal`) is not duplicated.
* Main `results/vanda/` and `figures/` are not duplicated.
* The report ZIP/PDF, personal report directories, personal QA/checklist files,
  workspace-separation notes and personal READMEs are not included.
* Legacy `1552672..._e2_strong_256_hetero.csv` and the invalid unpinned
  256-core rows are excluded from valid benchmark evidence.
* Invalid/obsolete campaign scripts, probes and diagnostics are excluded.
* Raw `data/covtype.data`, binaries, `__pycache__`, build logs and credentials
  are excluded.

## Dependency notes

* The supplementary C++ differs from main only by fixed-total resampling
  (`--resampled-total`), generated-workload bookkeeping and the matching test.
  It is required for the 10M fixed-total runs and is isolated.
* `bencher.py` is copied into `supplementary/scripts/` so `bencher.ROOT`
  resolves to `supplementary/` and the standalone runner works without the main
  tree's result paths.
* PBS `common.sh` contains no account/scratch path; `#PBS -P YOUR_ALLOCATION`
  must be replaced by the submitter. Submit from `supplementary/`.

## Validation results

* Serial build with `g++ -O3 -std=c++17 -DNDEBUG` succeeded.
* Serial 20k run succeeded: hash `8777036785045208508`.
* Fixed-total 200,003 run succeeded: hash `18302656097830585624`, rows
  200,003; this confirms the resampling path compiles and runs.
* Python files `py_compile` clean; all PBS scripts pass `bash -n`.
* Full OpenMP build on this macOS host failed because `omp.h`/libomp is not
  installed; this is a local-host prerequisite, not a source error. MPI build
  was not attempted because no `mpicxx` is installed locally.
* Vanda execution was not rerun. The committed raw CSVs and logs are the
  evidence; no measured values were regenerated.
* `validation/checksums.sha256` records SHA-256 for all 13 CSVs using relative
  paths.
* `validation/summary.csv` aggregates 69 measured groups.

## Report traceability

The requested `DSA5208_Proj3_Report_v7.pdf` and
`DSA5208_Proj3_Overleaf_v7.zip` were absent at integration time. The available
v6 package was inspected without modification. Mapping is recorded in
`supplementary/validation/REPORT_TRACEABILITY.md`; every supplementary
experiment has a CSV except E6-128, which is explicitly pending.

## Known limitations

* E6-128 has no completed result; 1557221/1563236 are queued and 1563237 is
  held.
* The 256-worker set mixes six unpinned layouts with a separate pinned 256x1
  recovery; it is not a single controlled binding experiment.
* E2 1-rank recovery is on a different node family from the primary 4x64 curve.
* E7 core/ppr32 placement printouts are absent; timing CSVs are complete.
* No raw dataset, credentials or SSH keys are included.
