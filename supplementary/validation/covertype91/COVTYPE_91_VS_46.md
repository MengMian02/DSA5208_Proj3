# Covertype 91-iteration vs 46-iteration runs

These are two distinct clustering outcomes. They must not be merged or treated
as replicates.

## 91 iterations (report result)

- Artifacts: `supplementary/validation/covertype91/`.
- Inertia 3,712,497.960091; label hash 9,039,984,004,050,467,668.
- Cluster sizes: 77,764; 56,307; 50,440; 51,673; 125,172; 66,197; 153,459.
- ARI 0.022010; NMI 0.064955; AMI 0.064936.
- No PBS job ID, binary hash, compiler/host record or verified exact command
  was preserved.

## 46 iterations (separate Vanda campaign)

The separate campaign is recorded on `repair-2026-11-14` (artifact commit `425c420`).
- Paths:
  - `results/vanda_harry/covtype_converged_1552669.stdct-mgmt-02/serial_converged.log`
  - `results/vanda_harry/covtype_converged_1552669.stdct-mgmt-02/centroids.csv`
  - `results/vanda_harry/covertype_metrics.json`
- PBS job 1552669, CN-161, exit 0.
- Documented command in `jobs/vanda/analysis_hc.pbs`:
  `./kmeans_serial data/covtype.data 7 300 42 --centroids <file>`.
- Inertia 3,722,934.178255; label hash 12,199,449,322,292,579,472.
- ARI 0.019122; NMI 0.056173; AMI 0.056154.

## Interpretation

The two runs use the same original dataset, K, D, seed, maximum iteration count
and convergence stopping rule, but produce different local optima and different
inertia. The preserved 91 artifact has weaker execution provenance than the 46
run. The exact cause of the difference (binary/source revision,
standard-library RNG/initialisation behaviour, input state, or another
undocumented factor) has not been established.

The report's 91-iteration result remains the reported result. The 46-iteration
run is retained as separate evidence only.
