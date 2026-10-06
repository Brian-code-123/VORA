#!/usr/bin/env bash
# Final quiet-host measurement run (T8). Quit Docker Desktop, browsers and other heavy apps first.
# Writes results/*.json (VORA_WRITE_RESULTS=1) and results/suites/*; log in results/final_measure.log.
set -uo pipefail
cd "$(dirname "$0")/.."
export VORA_WRITE_RESULTS=1 PYTHONPATH=src:.
PY=.venv/bin/python
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a results/final_measure.log; }
quiet_wait() { for i in $(seq 1 60); do $PY -c "from vora.hostcheck import is_quiet; import sys; sys.exit(0 if is_quiet() else 1)" && return 0; sleep 10; done; log "host never quiet"; }
: > results/final_measure.log
log "start"
quiet_wait; log "perf tests (memory, TTS first chunk)"
$PY -m pytest -m perf tests/test_memory.py tests/test_tts.py -q --timeout=900 2>&1 | tail -3 | tee -a results/final_measure.log
quiet_wait; log "bench (synthetic en/zh, 42 unique questions, speculation default)"
$PY scripts/bench.py --audio-only --out results/bench.json 2>&1 | tail -2 | tee -a results/final_measure.log
quiet_wait; log "2 concurrent users"
$PY scripts/bench_concurrent.py 2>&1 | tail -2 | tee -a results/final_measure.log
for s in minds14_en_us minds14_en_gb minds14_en_au minds14_zh; do
  quiet_wait; log "real voices: $s (test split, ASR + retrieval + faithfulness + real-time latency)"
  $PY scripts/eval_suite.py --suite "$s" --split test --final --n 100 --kb bank --faith --latency 2>&1 | grep -v -i "warn\|re_" | tail -3 | tee -a results/final_measure.log
done
quiet_wait; log "ASR eval (LibriSpeech clean/other, FLEURS, AISHELL)"
$PY scripts/eval_asr.py --n 50 2>&1 | tail -3 | tee -a results/final_measure.log
log "scene matrix (test split, 100 clips per cell)"
$PY scripts/run_matrix.py --split test --final --n 100 --out results/suites_final.json 2>&1 | grep -v -i "warn\|re_" | tee -a results/final_measure.log
$PY scripts/report_gates.py --write results/gates_final.json 2>&1 | tee -a results/final_measure.log
log "done"
