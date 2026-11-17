# Report traceability

The requested `DSA5208_Proj3_Report_v7.pdf` and
`DSA5208_Proj3_Overleaf_v7.zip` were **not present** in the parent directory at
integration time. The latest available package was v6; it was inspected without
modification. The supplementary mapping below uses the report's experiment
names and the CSV files in this directory.

| Report experiment | CSV/source |
|---|---|
| 10M fixed-total MPI strong | `results/1552703..._e2_large.csv` |
| 581k personal MPI strong (replication) | `results/1557218..._e2_strong.csv` |
| E2 1-rank recovery | `results/1563234..._e2_rank1_recovery.csv` |
| Vanda OpenMP 581k/10M | `results/1557219..._e5_openmp_*.csv` |
| Hybrid 64 (10M) | `results/1557220..._e6_hybrid_64.csv` |
| Hybrid 256 six unpinned layouts | `results/1557222..._e6_hybrid_256.csv` |
| Hybrid 256x1 pinned recovery | `results/1563235..._e6_256_rank1_recovery.csv` |
| E7 placement | `results/1557223..._e7_{core,ppr32,ppr16}.csv` |
| E6-128 | pending; no result file |

No measured value was regenerated or substituted. Report v7 verification is a
blocker for the report packaging task, not for this integration branch.
