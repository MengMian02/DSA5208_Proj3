# Report-to-source traceability

## Reference report

The newest actual report available is:

- `DSA5208_Proj3_Report_v6.pdf` — 20 pages, 749,232 bytes, timestamp
  2026-11-17 13:37.
- `DSA5208_Proj3_Overleaf_v6.zip` — 30 entries, timestamp 2026-11-17 13:38.

No `DSA5208_Proj3_Report_v7.pdf` or `DSA5208_Proj3_Overleaf_v7.zip` exists in
the project directory. The v6 report is therefore the traceability reference.
The report PDF/ZIP are not committed; this file records only source links.

## Mapping

| Report element | Source evidence | Code revision | Generator / output | Status |
|---|---|---|---|---|
| §6 serial Tables 2–3, Fig. 4 | `results/vanda/vanda_e1_serial_1546646.stdct-mgmt-02.csv` | c450025 | `validation/analysis/legacy_make_report_assets.py`; `tables/serial_N.tex`, `serial_K.tex`, `serial.pdf` | Complete; report generator is a preserved legacy script, not part of main code |
| §7 MPI strong Table 4, Fig. 5 | `results/vanda/vanda_e2_strong_1551056...csv`, `...8x32_1551069...csv` | c450025 | legacy report asset generator; `strong.pdf` | Complete |
| §7 10M strong Fig. 6 | `supplementary/results/1552703...e2_large.csv` | ab64b19 | report `strong_10m.pdf`; generator not committed in main/report archive | Source complete; figure generator link partial |
| §8 MPI weak Table 5, Fig. 7 | `results/vanda/vanda_e3_weak_1551057...csv`, `...8x32_1551070...csv` | c450025 | legacy report asset generator; `weak.pdf` | Complete |
| §9 Vanda OpenMP Table 6, Fig. 8 | `supplementary/results/1557219...e5_openmp_original.csv`, `...e5_openmp_10m.csv` | ab64b19 | report `openmp_vanda.pdf`; generator not committed | Source complete; generator link partial |
| §9 M1 OpenMP Table 7, Fig. 9 | `results/e4_hybrid.csv` (Apple M1 rows) | c450025 | report `openmp.pdf`, `tables/openmp.tex` | Complete, separate platform |
| §10 Hybrid 64 Table 8, Fig. 10 | `results/vanda/vanda_e4_hybrid72_1551076...csv` | c450025 | legacy report asset generator; `hybrid64.pdf` | Complete |
| §10 256 Hybrid Table 9, Fig. 11 | `results/vanda/e4_pinned_4x64_1556850.csv`; invalid `e4_hybrid256_1554129`/`e4_hybrid_8x32_1554130` diagnostic only | c450025 | report `hybrid256.pdf` | Pinned valid; unpinned invalid/diagnostic |
| §10 10M Hybrid Table 10, Fig. 12 | `supplementary/results/1557222...e6_hybrid_256.csv` (six unpinned) and `supplementary/results/1563235...e6_256_rank1_recovery.csv` (separate pinned) | ab64b19 | report `e6_256_recovery.pdf`; generator not committed | Source complete; binding/allocation distinction explicit |
| §11 CUDA Table 11, Fig. 13 | `results/vanda/e7_cuda_1554475.csv` | c450025 | legacy report asset generator; `cuda.pdf` | Complete |
| §11 Metal Fig. 14 | `results/e6_metal.csv` | c450025 | report `metal.png` | Complete, separate platform |
| §12 communication Fig. 15 | `results/vanda/vanda_e2_strong_*` | c450025 | legacy report asset generator; `communication.pdf` | Complete |
| §12 E7 placement | `supplementary/results/1557223...e7_core.csv`, `...ppr32.csv`, `...ppr16.csv` | ab64b19 | narrative; no separate table/figure | Timing complete; host printout limited |
| §13 correctness | E2/E4/E5/E6/E7 CSVs; hashes/SSE | mixed | tests and CSV columns | Cross-evidence present |
| §15 Covertype Table 13, Figs. 16–17 | `supplementary/validation/covertype91/covtype_converged.log`, centroids, metrics, clusters, contingency | 91-run provenance incomplete | `covertype91/analyze_covertype.py`; table/figures in report v6 | Evidence preserved; original execution chain incomplete |

## Boundaries

- Core baseline CSVs and supplementary CSVs are never merged into a single
  speedup curve.
- The six unpinned 256-worker layouts and the separately allocated pinned
  256×1 recovery are kept separate.
- The 91-iteration Covertype result is used by the report; the separate
  46-iteration Vanda result is not substituted.
- `strong_10m.pdf`, `openmp_vanda.pdf` and `e6_256_recovery.pdf` exist in the
  report package, but their generation scripts were not all committed. The
  source CSVs and report outputs are preserved; this is a partial generator
  link, not a missing measurement.
