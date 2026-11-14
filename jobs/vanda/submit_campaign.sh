#!/bin/bash
# Submit the repaired E0-gated campaign exactly once.
# Safe to rerun: jobs already present in qstat -u harry_chen are skipped.
set -u
set -o pipefail

cd "$(dirname "$0")/../.." || exit 1
WORKDIR=$(pwd)
LOG_DIR="$WORKDIR/results/vanda_harry"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%dT%H%M%S)
RECORD="$LOG_DIR/submission_${STAMP}.log"

if ! timeout 20 qstat -u harry_chen >/dev/null 2>&1; then
  echo "ERROR: PBS is not responding; refusing to submit." >&2
  echo "Retry when: qstat -u harry_chen" >&2
  exit 2
fi

existing=$(qstat -u harry_chen 2>/dev/null)
submit() {
  name="$1"; script="$2"; depend="${3:-}"
  if printf '%s\n' "$existing" | grep -q "$name"; then
    echo "SKIP $name (already in queue)" | tee -a "$RECORD"
    return 0
  fi
  if [ -n "$depend" ]; then
    out=$(qsub -P personal-harry_chen -W "depend=afterok:${depend}" "$script" 2>&1)
  else
    out=$(qsub -P personal-harry_chen "$script" 2>&1)
  fi
  status=$?
  printf '%s: %s\n' "$name" "$out" | tee -a "$RECORD" >&2
  if [ $status -ne 0 ]; then
    echo "ERROR submitting $name" >&2
    return 1
  fi
  id=$(printf '%s\n' "$out" | grep -oE '^[0-9]+' | head -1)
  [ -n "$id" ] || { echo "ERROR: no job id for $name" >&2; return 1; }
  printf '%s=%s\n' "$name" "$id" >> "$RECORD.ids"
  printf '%s\n' "$id"
}

e0=$(submit E0 jobs/vanda/e0_gate_hc.pbs) || exit 1
echo "E0 job id: $e0"
submit E2 jobs/vanda/e2_strong_hc.pbs "$e0" >/dev/null || exit 1
submit E5 jobs/vanda/e5_openmp_hc.pbs "$e0" >/dev/null || exit 1
submit E6-64 jobs/vanda/e6_hybrid_64_hc.pbs "$e0" >/dev/null || exit 1
submit E6-128 jobs/vanda/e6_hybrid_128_hc.pbs "$e0" >/dev/null || exit 1
submit E6-256 jobs/vanda/e6_hybrid_256_hc.pbs "$e0" >/dev/null || exit 1
submit E7 jobs/vanda/e7_layout_hc.pbs "$e0" >/dev/null || exit 1

echo "Submission record: $RECORD"
echo "Job ids:"
cat "$RECORD.ids" 2>/dev/null
qstat -u harry_chen
