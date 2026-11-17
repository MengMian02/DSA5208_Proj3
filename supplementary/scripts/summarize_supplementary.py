#!/usr/bin/env python3
"""Summarise the supplementary CSVs without changing measured values.

Groups rows by experiment/implementation/ranks/threads/rows/dataset, reports the
median of each numeric timing field, and checks that all repetitions share one
label hash. Writes validation/summary.csv.
"""
import csv, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "validation" / "summary.csv"
FIELDS = ["fit_s", "mpi_reduction_s", "local_compute_s", "thread_reduction_s",
          "sse", "total_s", "obs_per_rank"]

def main():
    groups = defaultdict(list)
    for path in sorted(RESULTS.glob("*.csv")):
        for row in csv.DictReader(path.open()):
            key = (row.get("experiment", ""), row.get("implementation", ""),
                   row.get("ranks", ""), row.get("threads", ""),
                   row.get("rows", ""), row.get("dataset", ""), path.name)
            groups[key].append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "implementation", "ranks", "threads", "rows",
                         "dataset", "source", "reps", "label_hashes", "fit_s_median",
                         "collective_s_median", "local_s_median", "sse_median"])
        for key, rows in sorted(groups.items()):
            def med(field):
                vals = [float(r[field]) for r in rows if r.get(field) not in (None, "")]
                return "%.6f" % statistics.median(vals) if vals else ""
            hashes = sorted({r.get("label_hash", "") for r in rows})
            writer.writerow(list(key) + [len(rows), ";".join(hashes), med("fit_s"),
                             med("mpi_reduction_s"), med("local_compute_s"), med("sse")])
    print("wrote", OUT, "groups", len(groups))

if __name__ == "__main__":
    main()
