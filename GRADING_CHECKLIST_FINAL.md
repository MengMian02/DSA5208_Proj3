# GRADING_CHECKLIST_FINAL — DSA5208 Project 3

Final report: `DSA5208_Proj3_Final_Report.pdf` (19 pages), source
`report_final/main.tex`. Evidence base combines GitHub `c450025` and personal
`3ef17dc` (local, not pushed by this task).

| Marking item | Points | Report evidence | Status |
|---|---:|---|---|
| Algorithm explanation | 5 | Section 3; pseudocode Figure 1; equations (1)–(7); complexity | Complete |
| Serial scaling | 5 | Section 6; Tables 3–5; Figure 2; GitHub 50k–581k + personal 1M/5M/10M | Complete |
| MPI strong scaling | 5 | Section 7; Tables 6–7; Figures 3–4; 581k and 10M to 256 ranks | Complete |
| MPI weak scaling | 5 | Section 8; Table 8; Figure 5; 50k/rank to 256 ranks | Complete |
| MPI+OpenMP and GPU | 10 | Sections 9–11; Tables 9–13; Figures 6–11; Vanda OpenMP 1–64, Hybrid 64/partial 256, pinned 256×1, CUDA A40, Metal | Complete, with E6 gaps flagged |
| Overall discussion | 5 | Section 15; collective latency, hardware confound, Hybrid/GPU trade-offs, limitations | Complete |
| Correctness validation | Required | Section 13; hashes for E2, E4, E5/E6, E3, pinned; CUDA/Metal agreement | Complete |
| Vanda runtime figures | Required | Sections 6–12; all runtime values from named Vanda CSVs | Complete |
| Parallelization design | Required | Section 4; MPI decomposition, two Allreduces, OpenMP fold, Hybrid, GPU | Complete |
| Scaling efficiency / communication | Required | Sections 7–8, 12; speedup, efficiency, collective share, E7 | Complete |
| Real-data Covertype conclusions | Required | Section 14; Table 15; Figures 12–13; ARI 0.0220, NMI 0.0650 | Complete |
| Reproducibility | Required | Section 18; build/run commands; CSV provenance; scripts | Complete |
| AI-use declaration | Required | Section 19; accurate statement including PBS submissions | Complete |
| Source code and README | Required | Submission ZIP; `README.md`, `README_SUBMISSION.md` | Complete |
| Team names and student IDs | Required | Title page placeholders | **Author action** |
| Provenance separation | Required | Table 2; `RESULT_PROVENANCE_FINAL.csv` | Complete |
| Pending recovery jobs | Required | Section 16; Table 16; `RECOVERY_STATUS.md` | Transparently pending |

## Previous report problems resolved

1. **Missing 256-core multi-thread Hybrid evidence:** added personal 10M
   256-core partial (18/21 rows, six layouts) with missing 256×1 explicitly
   flagged; recovery 1563235 queued. No fake completion.
2. **Overly strong CPU-pinning causal claims:** wording now says the pinned
   experiment controls binding only and does not prove which of binding, NUMA
   or competing threads caused the old spread.
3. **Missing Vanda OpenMP scaling:** added E5 1–64 threads on original and 10M
   data (42 rows); replaces the M1-only curve as primary.
4. **Misinterpretation of per-rank phase timers:** report states maxima are not
   additive and never sums them; Allreduce includes waiting.
5. **Incomplete/inaccurate AI declaration:** declaration now records PBS
   queries, downloads, recovery/backup submissions and no cancellations/push.
6. **CUDA kernel-vs-fit misuse:** kernel 0.365 ms is kept separate from the
   0.644 s full fit and 89.5 ms CPU assignment.
7. **10M strong scaling:** added 27-row personal campaign, 133.0× / 52.0% at
   256 ranks.
8. **Vanda OpenMP only up to 64 threads:** 128–256 thread limitation stated.

## Invalid/excluded evidence

- Unpinned 256-core Hybrid 1554129/1554130: INVALID, diagnostic only.
- E2 1557218 missing 1-rank: PARTIAL; recovery 1563234 reported standalone.
- E6-256 1557222 missing 256×1: PARTIAL; recovery pending.
- E6-128 1557221: QUEUED; no result claimed.
- E6-128 backup 1563236/1563237: PENDING; no result claimed.
- Old Atlas results: historical context only, not mixed into Vanda curves.
