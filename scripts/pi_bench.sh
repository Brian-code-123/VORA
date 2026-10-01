#!/usr/bin/env bash
# Run on a real Raspberry Pi 4 (64-bit OS) after `scripts/deploy_pi.sh` or a manual install. Writes results/pi_*.json.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:.
echo "host: $(uname -m), $(nproc) cores, $(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo) MB RAM"
python scripts/bench.py --audio-only --out results/pi_bench.json "$@"
python scripts/eval_asr.py --n 30
python scripts/report_gates.py --write results/gates_pi.json
