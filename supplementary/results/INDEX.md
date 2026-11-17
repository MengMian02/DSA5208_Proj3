# Supplementary result index

| File | Experiment | Rows | Role |
|---|---|---:|---|
| `1552578..._e1_serial.csv` | E1 serial + 1M/5M/10M | 39 | serial extension |
| `1552703..._e2_large.csv` | E3 fixed 10M MPI strong | 27 | 1--256 rank strong scaling |
| `1552672..._e3_weak_256_hetero.csv` | E4 weak scaling | 27 | 50k observations/rank |
| `1557218..._e2_strong.csv` | E2 personal 581k strong | 24 | 2--256 ranks; 1-rank missing |
| `1563234..._e2_rank1_recovery.csv` | E2 1-rank recovery | 3 | supplementary 1-rank evidence |
| `1557219..._e5_openmp_original.csv` | E5 OpenMP 581k | 21 | 1--64 threads |
| `1557219..._e5_openmp_10m.csv` | E5 OpenMP 10M | 21 | 1--64 threads |
| `1557220..._e6_hybrid_64.csv` | E6 Hybrid 64 | 21 | seven layouts, 10M |
| `1557222..._e6_hybrid_256.csv` | E6 Hybrid 256 partial | 18 | six unpinned layouts, 10M |
| `1563235..._e6_256_rank1_recovery.csv` | E6 256x1 recovery | 3 | separate pinned allocation |
| `1557223..._e7_core.csv` | E7 placement | 4 | 128/256 ranks |
| `1557223..._e7_ppr32.csv` | E7 placement | 4 | 128/256 ranks |
| `1557223..._e7_ppr16.csv` | E7 placement | 2 | 128 ranks |

No raw `covtype.data` is included. All CSVs are byte-preserved from the
personal campaign; checksums are in `validation/checksums.sha256`.
