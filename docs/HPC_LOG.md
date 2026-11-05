# HPC log

Append-only record of what was actually submitted and observed. One entry per
job. Nothing is estimated: if a value was not observed, it stays blank.

Status: **no job has been submitted to Atlas or Vanda yet.**

## Required fields per entry

    date:
    cluster:                  Atlas | Vanda
    scheduler job id:
    probe file:               jobs/probe_<host>.txt
    allocation:               nodes x cores_per_node = total cores
    layouts:                  e.g. 4x64 (ranks x threads)
    walltime requested:
    queue wait observed:      from the scheduler record, not estimated
    walltime used:            from the scheduler record
    exit state:
    results file:             results/<name>.csv
    placement evidence:       what the launcher actually bound (paste the line)
    notes:

## Checkpoint sequence before production submission

1. Fill jobs/cluster.env from a compute-node probe.
2. Submit one tiny job: 2 ranks, 5 iterations, --rows 20000. Confirm it starts,
   writes one csv row, and the 2-rank label_hash equals the serial label_hash
   for the same subset.
3. Submit the strong-scaling job for the full rank sweep.
4. Submit the weak-scaling job (source binary cached once first).
5. Submit the 72-core hybrid layouts and one multi-node layout.
6. Copy result CSVs back, rerun scripts/analyze.py, and record here which
   figure each job fed.

## Known open questions to settle on the cluster

* Whether the site launcher accepts --bind-to core --map-by core (or
  --cpu-bind=cores) and what it actually binds. If unsupported, record that the
  E2/E3 point is unpinned rather than assuming binding.
* Physical cores per node, and whether hyperthreads are exposed to the job. More
  ranks than physical cores would silently turn E2 into an oversubscription
  experiment.
* Whether the parallel filesystem handles 256 ranks reading the 252 MB source
  binary concurrently; if not, use the broadcast path and record the cost.
* Whether walltime limits force splitting the rank sweep across several jobs.
  Splitting is acceptable but then each job must record its own node set.
