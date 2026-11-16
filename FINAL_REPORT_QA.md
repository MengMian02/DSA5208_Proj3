# FINAL_REPORT_QA

Date: 2026-11-16. Final report baseline: GitHub `c450025` plus personal
`3ef17dc` and the local recovery/report work.

## Compilation

* `report_final/main.tex` compiled independently in
  `/tmp/DSA5208_Final_Overleaf` with
  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.
* Result: **19 pages**, 633 KB PDF.
* Undefined citations/references: **0**.
* Overfull boxes: **0**.
* Bibliography: 9 cited entries.
* All figures/tables are present; no placeholder or missing asset.

## Numerical checks

| Claim | Check |
|---|---|
| 10M strong: 49.805 s → 0.374 s, 133.0×, 52.0% | personal E3 27 rows |
| 581k strong: 2.887 s → 0.0223 s, 129.7×, 50.7% | GitHub E2 4×64 27 rows |
| Weak 4×64: 80.2% at 256 | GitHub E3 27 rows |
| Vanda OpenMP: 2.530 → 0.1045 s, 24.2× at 64 | personal E5 21 rows |
| E6-64 10M: 1.257 s (64×1) vs 1.735 s (1×64) | personal E6 21 rows |
| E6-256 partial: 18/21, no 256×1 | personal E6-256 18 rows |
| E7 8×32: 256 rank fit 0.480–0.526 s, collective 0.467–0.514 s | personal E7 10 rows |
| E2 1-rank recovery: median 3.429 s, 3 rows, hash same | job 1563234 |
| CUDA: 0.644 s full fit, 0.365 ms kernel, 89.5 ms CPU assign | GitHub CUDA 30 rows |
| Covertype: ARI 0.0220, NMI 0.0650, inertia 3,712,497.96 | baseline E9 analysis |

## Provenance and exclusions

* `RESULT_PROVENANCE_FINAL.csv` maps all 20 result series to source, file,
  rows, status and report section.
* GitHub and personal results are kept separate; different datasets, node
  families and binding modes are not merged into one curve.
* Invalid unpinned 256-core Hybrid rows 1554129/1554130 are excluded from
  performance conclusions.
* E2 1557218 is labelled PARTIAL; the 1563234 recovery is standalone because it
  ran on IN-010 rather than the original mixed CN/GN allocation.
* E6-256 1557222 is labelled PARTIAL; 256×1 is missing.
* E6-128 1557221 is queued; backup 1563236/1563237 are pending/held. No result
  is claimed for them.
* No raw `data/covtype.data`, compiled binary, credential, SSH key or Git
  metadata is included in the submission package.
* Team names/student IDs remain placeholders; authors must fill them.

## Previous-problem checks

1. 256-core multi-thread Hybrid: personal 10M partial table/figure added; gap
   explicit; no fabricated 256×1.
2. Pinning causation: report states binding controls only one factor and does
   not prove sole cause.
3. Vanda OpenMP: E5 1–64 thread sweep added as primary; M1 removed as primary.
4. Phase timers: report states maxima are not additive; no sum.
5. AI declaration: now records downloads and recovery/backup submissions, no
   cancellations or pushes.
6. CUDA kernel/fit: kernel, assignment, accumulation and full fit separate.
7. 10M strong scaling: 27-row personal campaign included.
8. Report limitations explicitly list E5 >64 threads, E2/E6 partials and
   pending E6-128.

## Package checks

* Final Overleaf ZIP (863 KB, 38 files) compiled independently; contains
  19-page PDF, editable `main.tex`, references, 14 report figures, 16 tables
  and analysis scripts.
* Final submission ZIP (4.8 MB, 261 files) contains report, C++/CUDA source,
  PBS scripts, tests, README, results, logs, provenance and checklists.
* ZIP extraction checked: 19-page `report_final/main.pdf`, no raw data, no
  `.git`, no LaTeX build `.aux`/`.log` files; relevant PBS/result logs are included; Overleaf extraction recompiled to 19 pages with
  0 undefined references and 0 overfull boxes.
