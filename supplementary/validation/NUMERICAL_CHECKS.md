# Numerical checks

* E2 1-rank recovery `1563234`: 3 rows, median fit 3.429333 s, label hash
  `15912193932519828622`, SSE 3,722,934.336211. This is supplementary because
  the node family differs from the primary 4x64 campaign.
* E6-256 256x1 recovery `1563235`: 3 rows, median fit 0.320140 s, label hash
  `14076111245358280480`, SSE 64,089,789.818295. Separate four-CN-node
  allocation with `--map-by core --bind-to core`.
* E6-256 original `1557222`: 18 rows (six unpinned layouts, 3 reps), one label
  hash; 256x1 missing from the original job.
* E6-64 `1557220`: 21 rows, seven layouts, 3 reps, one label hash.
* E5 `1557219`: two 21-row sweeps (581k and 10M), one hash per workload.
* E7 `1557223`: 10 rows across core/ppr32/ppr16; each file has one hash.
* E2 personal `1557218`: 24 rows, ranks 2--256, one hash; 1-rank failed.
* E6-128: no completed rows. Jobs 1557221/1563236/1563237 are pending/held;
  no hash or SSE is claimed.

Checksums: `validation/checksums.sha256`.
