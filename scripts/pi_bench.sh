#!/usr/bin/env bash
# Pi-class measurement inside the VORA image (Raspberry Pi 4, AWS a1.xlarge / Cortex-A72, or any arm64 box):
#   docker run --rm -v "$PWD/results-pi:/app/results" vora:arm64 bash scripts/pi_bench.sh
# Writes results/pi_cpu.txt, results/pi_bench.json (real-time paced turns, audio mode), results/pi_tts.json, results/pi_mem.json (USS per component).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:.
mkdir -p results
{ echo "arch: $(uname -m)"; echo "cores: $(nproc)"; awk '/MemTotal/ {print "mem_mb: " int($2/1024)}' /proc/meminfo
  (lscpu 2>/dev/null || cat /proc/cpuinfo) | grep -Ei "model name|cpu part|bogomips|mhz" | sort -u | head -8; } | tee results/pi_cpu.txt
python - <<'PY'
import json, time, numpy as np
from vora.config import Settings
from vora.tts import Tts
from vora.hostcheck import uss_mb
t0 = time.perf_counter(); tts = Tts(Settings()); load = time.perf_counter() - t0
list(tts.synth("Hello.")); list(tts.synth("你好。"))
out = {"load_s": round(load, 2)}
for lang, words in (("en", ["The", "Yes", "Hold", "Please", "Sure", "Two", "Your", "It", "Blue", "Reset"]), ("zh", ["保修", "你好", "按住", "可以", "蓝色", "两年", "请", "设备", "支持", "固件"])):
    t = []
    for w in words * 2:
        s = time.perf_counter(); next(iter(tts.synth(w + " "))); t.append((time.perf_counter() - s) * 1000)
    clause = "The warranty on the X200 is two years." if lang == "en" else "X200的保修期是两年。"
    s = time.perf_counter(); n = sum(len(c) for c in tts.synth(clause)) / 2
    out[lang] = {"first_chunk_p50_ms": round(float(np.percentile(t, 50))), "first_chunk_p95_ms": round(float(np.percentile(t, 95))),
                 "rtf": round((time.perf_counter() - s) / (n / 16000), 3)}
out["uss_mb_tts"] = round(uss_mb())
json.dump(out, open("results/pi_tts.json", "w"), indent=1); print(out)
PY
python scripts/bench.py --audio-only --force --n 20 --out results/pi_bench.json
python scripts/mem_profile.py --out results/pi_mem.json || true
echo "done: results/pi_*"
