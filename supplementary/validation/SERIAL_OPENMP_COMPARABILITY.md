# Serial vs OpenMP one-thread comparability

## Measured records

| Field | Serial E1 | OpenMP E5, one thread |
|---|---|---|
| CSV | `results/vanda/vanda_e1_serial_1546646.stdct-mgmt-02.csv` | `supplementary/results/1557219.stdct-mgmt-02_e5_openmp_original.csv` |
| Job | 1546646 | 1557219 |
| Host | IN-002 | IN-012 |
| CPU model | Intel Xeon Gold 6438N | Intel Xeon Gold 6438N |
| Date | 2026-11-12 | 2026-11-15 |
| Account/allocation | team Vanda account | supplementary Vanda account |
| Source revision | main `c450025` | supplementary `ab64b19`/`425c420` |
| Compiler | GCC 11.4.1 | GCC 11.4.1 |
| Flags | `-O3 -std=c++17 -DNDEBUG` | `-O3 -std=c++17 -DNDEBUG` |
| Iterations | 30 fixed | 30 fixed |
| Rows | 581,012 | 581,012 |
| Median fit | 3.380733 s | 2.530182 s |
| Label hash | `15912193932519828622` | `15912193932519828622` |
| SSE | `3722934.336211` | `3722934.336211` |

Relative difference:

- `(3.380733 - 2.530182) / 3.380733 = 25.16%`
- `(3.380733 - 2.530182) / 2.530182 = 33.62%`

## Why this is not a controlled comparison

The records agree on data, row count, K, D, seed, iteration count, label hash
and SSE, but differ in:

- physical node (IN-002 vs IN-012);
- date and allocation/account;
- source revision and compiled binary;
- serial fused loop vs OpenMP thread-local accumulation and reduction;
- affinity policy: the E5 preflight sets `OMP_PROC_BIND=close` and
  `OMP_PLACES=cores`, while the E1 serial job has no equivalent OpenMP
  affinity control.

No host load, CPU-frequency or memory-bandwidth telemetry was preserved. The
available evidence therefore cannot attribute the 25% difference to OpenMP,
compiler optimisation, CPU affinity, node state or source revision.

## Reporting rule

Both measured values remain valid for their own experiments. OpenMP speedup is
calculated against its own one-thread baseline, not against the serial
absolute runtime. The report must not claim that OpenMP is intrinsically 25%
faster than Serial on the basis of these two records.

No rerun was performed for this repair.
