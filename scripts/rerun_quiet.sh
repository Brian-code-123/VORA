#!/usr/bin/env bash
# Re-measure real-voice suites (test split, --final) only on a quiet host, then refresh results/gates_final.json.
# Usage: bash scripts/rerun_quiet.sh [suite ...]        (default: the four MInDS-14 suites)
# Waits for 3 quiet checks in a row (QUIET_POLL_S apart). Exit 2: never quiet within QUIET_WAIT_MIN minutes (prints what
# is using the CPU). Exit 1: a suite stayed busy for 3 tries; its gate stays UNVERIFIED. The quiet threshold is never
# lowered and VORA_FORCE_PERF is never set here. Close other apps and keep the browser pane shut while it runs.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
QUIET_CHECK=${QUIET_CHECK:-"PYTHONPATH=src $PY -c 'import sys; from vora.hostcheck import perf_skip_reason as r; sys.exit(1 if r() else 0)'"}
QUIET_WAIT_MIN=${QUIET_WAIT_MIN:-60}
QUIET_POLL_S=${QUIET_POLL_S:-20}
RESULTS_DIR=${RESULTS_DIR:-results/suites}
SUITES=("$@")
[ ${#SUITES[@]} -eq 0 ] && SUITES=(minds14_en_us minds14_en_gb minds14_en_au minds14_zh)

wait_quiet() {
  local ok=0 deadline=$(( $(date +%s) + QUIET_WAIT_MIN * 60 ))
  while [ $ok -lt 3 ]; do
    if eval "$QUIET_CHECK"; then ok=$((ok + 1)); else ok=0; fi
    [ $ok -ge 3 ] && return 0
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "host never quiet in ${QUIET_WAIT_MIN} min; top CPU processes:"
      ps -Ao pcpu,comm -r | head -4
      exit 2
    fi
    sleep "$QUIET_POLL_S"
  done
}

is_quiet_result() {   # the result file exists, was measured quietly and carries latency
  "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('quiet') and d.get('latency') else 1)" "$1" 2>/dev/null
}

failed=()
for s in "${SUITES[@]}"; do
  ok=0
  for t in 1 2 3; do
    wait_quiet
    echo "[$(date +%T)] $s try $t"
    if [ -n "${SUITE_CMD:-}" ]; then eval "$SUITE_CMD"
    else VORA_WRITE_RESULTS=1 "$PY" scripts/eval_suite.py --suite "$s" --split test --final --n 100 --kb bank --faith --latency >/dev/null 2>&1
    fi
    if is_quiet_result "$RESULTS_DIR/${s}__clean__test.json"; then ok=1; break; fi
  done
  [ $ok -eq 1 ] || failed+=("$s")
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "no quiet result: ${failed[*]} (gate stays UNVERIFIED; see the .busy.json files)"
  exit 1
fi
[ -n "${SUITE_CMD:-}" ] || "$PY" scripts/report_gates.py --write results/gates_final.json
echo "all quiet: ${SUITES[*]}"
