"""Endpoint wait (speech end -> final) and extra-final rate for different trailing-silence rules, hold on/off.
Usage: eval_endpoint.py [--out results/endpoint.json]"""
import argparse
import json
from pathlib import Path

import numpy as np

import scripts.eval_asr as E
from vora.asr import AsrSession, load_recognizers
from vora.config import ROOT, Settings

SR = 16000
SETS = ("en_librispeech_clean", "en_fleurs", "zh_aishell")


def run_set(name: str, settings: Settings, hold_ms: int, n: int = 50) -> dict:
    lang = E.SETS[name][2]
    recs = load_recognizers(settings)
    waits, extra, clips = [], 0, 0
    for _ref, pcm in E.load(name, n):
        sess = AsrSession(recs, lang, hold_ms=hold_ms, hold_total_cap_ms=settings.hold_total_cap_ms)
        x = np.concatenate([pcm, np.zeros(int(2.5 * SR), dtype=np.int16)])
        t_first, finals = None, 0
        for i in range(0, len(x), 1600):
            for e in sess.feed(x[i:i + 1600].tobytes()):
                if e.kind == "final":
                    finals += 1
                    if t_first is None:
                        t_first = (i + 1600) / SR
        clips += 1
        extra += max(0, finals - 1)
        if finals:
            waits.append(max(0.0, ((finals and t_first) - len(pcm) / SR)) * 1000 if finals == 1 else float("nan"))
    w = [v for v in waits if v == v]
    return {"clips": clips, "single_final_clips": len(w), "wait_ms_p50": round(float(np.percentile(w, 50)), 0), "wait_ms_p95": round(float(np.percentile(w, 95)), 0),
            "extra_finals_per_clip": round(extra / clips, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "endpoint.json"))
    a = ap.parse_args()
    res = {}
    for rule2 in (0.4, 0.3):
        for hold in (0, 500):
            s = Settings(endpoint_rule2_s=rule2)
            for name in SETS:
                res[f"rule2={rule2}/hold={hold}/{name}"] = r = run_set(name, s, hold)
                print(f"rule2={rule2} hold={hold} {name}: {r}")
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
