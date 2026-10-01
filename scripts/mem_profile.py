"""Incremental USS profile in ONE fresh process: what each import / model adds. Usage: mem_profile.py [--out results/mem_profile.json]"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "mem_profile.json"))
    a = ap.parse_args()
    p = psutil.Process()
    uss = lambda: p.memory_full_info().uss / 2**20
    steps, last = {}, uss()
    base = last

    def mark(name: str) -> None:
        nonlocal last
        gc.collect()
        now = uss()
        steps[name] = round(now - last, 1)
        last = now

    for mod in ("numpy", "scipy.signal", "onnxruntime", "sherpa_onnx", "faiss", "jieba", "rank_bm25", "tokenizers", "fastembed", "llama_cpp"):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            steps[f"import {mod}"] = None
            continue
        mark(f"import {mod}")
    from vora.config import Settings
    S = Settings()
    from vora.asr import AsrSession, load_recognizers
    recs = load_recognizers(S)
    r = recs if isinstance(recs, dict) else None
    for lang in ("en", "zh"):
        AsrSession(recs, lang).feed(b"\x00\x00" * 1600)
        mark(f"asr {lang}")
    from vora.rag.retriever import Retriever
    ret = Retriever(S)
    ret.search("warranty")
    mark("rag (embedder + index + bm25)")
    from vora.tts import Tts
    t = Tts(S)
    list(t.synth("Hello there."))
    mark("tts en")
    list(t.synth("你好。"))
    mark("tts zh")
    total_no_llm = round(last - base, 1)
    from vora.llm import Llm
    from vora.rag.store import Hit
    l = Llm(S)
    list(l.stream("hi", [Hit("w", "Hello.", 1.0)]))
    mark("llm")
    out = {"metric": "USS", "base_mb": round(base, 1), "steps_mb": steps, "asr_rag_tts_total_mb": total_no_llm,
           "load1": __import__("os").getloadavg()[0], "ts": time.time()}
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
