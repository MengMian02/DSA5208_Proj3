# Covertype 91-iteration evidence provenance

## Purpose

These files preserve the evidence behind the report's **91-iteration**
Covertype result. They are copied byte-for-byte from the legacy report
analysis directory; no numerical artifact was regenerated or edited.

## Files and hashes

| File | Legacy origin | SHA-256 |
|---|---|---|
| `covtype_converged.log` | `report/analysis/covtype_converged.log` | `95c865dc84e0811c405bf955be9ecfebf32d5a376db37bfe047a5000ecb01074` |
| `covtype_centroids.csv` | `report/analysis/covtype_centroids.csv` | `cc4bde7c39f69c81375a1550e3a27a3a112fb398a970cbe38b505190525753ee` |
| `covertype_metrics.json` | `report/analysis/covertype_metrics.json` | `5e14a0d6fe3f8bb232edf7b7f2c5cdcb96f48ca4cf4251fc25723fa7fd1f6e3c` |
| `covertype_clusters.csv` | `report/analysis/covertype_clusters.csv` | `88e9f8bbd0376f2b3b4baa75af2b4ddebddd478c576605e5b782b25732d8d0fd` |
| `covertype_contingency.csv` | `report/analysis/covertype_contingency.csv` | `5757760887034ab5dee8b36db33ac7bf4d14fef023f015516706dc2feb7d5c0a` |
| `analyze_covertype.py` | `report/analysis/analyze_covertype.py` | `bd81aaf080907dfbbda140165217c04e39ad412d08492a647e628146d7b48fcf` |

The copied files are byte-identical to those legacy origins.

## Dataset and run configuration

- UCI Covertype original data, 581,012 observations.
- D=54 predictors: first 10 quantitative features z-scored with population
  standard deviation; remaining 44 binary indicators unchanged.
- K=7, seed=42, max_iter=300, convergence stopping.
- 91 iterations, converged=true.
- Inertia: 3,712,497.960091.
- Cluster sizes: 77,764; 56,307; 50,440; 51,673; 125,172; 66,197; 153,459.
- Label hash: 9,039,984,004,050,467,668.

## Report mapping

- Report v6, Section 15, page 17: Table 13 and the inertia/ARI/NMI/AMI text.
- Report v6, Figure 16 (page 17) and Figure 17 (page 18).
- The table and figures are byte-identical to the legacy report assets used to
  produce v6.

## Execution provenance

Documented here:

- the serial command shape used by the separate E9 job
  (`./kmeans_serial data/covtype.data 7 300 42 --centroids <file>`), which is
  **not** evidence of the exact 91-iteration command;
- the 91-iteration output and cluster metrics;
- the centroid matrix and analysis script used to derive labels and figures.

**Not available:**

- no PBS Job ID for the 91-iteration run;
- no preserved binary or binary hash;
- no recorded compiler, host, build flags or exact executed command;
- no saved raw label vector (labels are reconstructed from the centroids by
  `analyze_covertype.py`).

The 91 run must therefore be described as **preserved evidence with incomplete
execution provenance**, not as a fully reproducible historical job.
