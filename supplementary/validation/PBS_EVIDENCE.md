# PBS execution evidence

| Job | Purpose | State | Evidence |
|---|---|---|---|
| 1557208 | E0 build/correctness gate | F, exit 0 | `km-hc-e0gate.o1557208`, `BUILD_INFO.*` |
| 1557218 | E2 personal 581k strong | F, exit 1 | 24 rows; 1-rank timeout in `km-hc-e2strong.o1557218` |
| 1557219 | E5 OpenMP | F, exit 0 | 42 rows in two CSVs; `km-hc-e5omp.o1557219` |
| 1557220 | E6-64 Hybrid | F, exit 0 | 21 rows; `km-hc-e6-64.o1557220` |
| 1557222 | E6-256 Hybrid | F, exit 1 | 18 rows; 256x1 preflight timeout |
| 1557223 | E7 placement | F, exit 0 | 10 rows; `km-hc-e7layout.o1557223` |
| 1563234 | E2 1-rank recovery | F, exit 0 | 3 rows; `km-hc-e2r1.o1563234` |
| 1563235 | E6-256 256x1 recovery | F, exit 0 | 3 rows; `km-hc-e6-256r1.o1563235` |
| 1557221 | E6-128 original | Q | no result |
| 1563236 | E6-128 backup smoke | Q | no result |
| 1563237 | E6-128 backup production | H (`afterok:1563236`) | no result |

The E6-256 recovery is a separate allocation from the six unpinned layouts.
The E6-128 jobs are reported as pending only.
