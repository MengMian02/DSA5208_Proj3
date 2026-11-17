# Failure and recovery notes

* **E2 1557218:** one-rank `mpirun --map-by core --bind-to core -np 1` timed
  out at 300 s inside a mixed four-node allocation, before application output;
  ranks 2--256 completed. Recovery 1563234 isolated the singleton on one
  IN-010 node with `--bind-to none` and completed 3/3 rows. The recovery is
  supplementary because node family differs.
* **E6-256 1557222:** one-rank preflight with
  `mpirun --map-by ppr:64:node --bind-to none -np 256` timed out at 90 s;
  six multi-thread layouts completed. Recovery 1563235 used
  `--map-by core --bind-to core` on four CN nodes and completed 3/3 rows.
  This does not prove binding was the sole cause: allocation and environment
  also changed.
* **E6-128:** 1557221 queued (`node_pool`); backup smoke 1563236 queued;
  backup production 1563237 held by `afterok:1563236`. No result is claimed.
* Invalid unpinned 256-core rows from the main repository (1554129/1554130)
  are not copied here as valid evidence.
