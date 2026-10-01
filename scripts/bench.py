"""Streaming latency / RTF / RSS benchmark on real models. Feeds audio at real-time pace (100 ms frames).
Usage: python scripts/bench.py [--n 30] [--out results/bench.json]"""
import argparse
import asyncio
import json
import os
import platform
import statistics
import time

import numpy as np
import psutil

from vora.config import ROOT, Settings
from vora.pipeline import Pipeline
from vora.rag.ingest import load_qa

SR = 16000
FRAME = 1600


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    lo, hi = int(np.floor(k)), int(np.ceil(k))
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def rtf(processing_s: float, audio_s: float) -> float:
    return processing_s / audio_s


def host() -> dict:
    return {"cpu": platform.processor() or platform.machine(), "machine": platform.machine(), "system": platform.platform(),
            "cores": psutil.cpu_count(logical=False), "loadavg_1m": round(os.getloadavg()[0], 1), "note": "Apple-Silicon Mac = arm64, NOT x86 and NOT a Raspberry Pi"}


def make_audio(tts, text: str) -> np.ndarray:
    return np.frombuffer(b"".join(tts.synth(text)), dtype=np.int16)


def answerable_questions() -> list[tuple[str, str]]:
    """All answerable eval questions, each once (no repeats: repeats flatter latency via cache hits)."""
    out = []
    for q in load_qa(ROOT / "eval" / "rag_qa.jsonl"):
        if q["chunk_id"]:
            out.append(("zh" if any("\u4e00" <= c <= "\u9fff" for c in q["q"]) else "en", q["q"]))
    return out


def questions(tts, n: int) -> list[tuple[str, str, np.ndarray]]:
    qa = [q for q in load_qa(ROOT / "eval" / "rag_qa.jsonl") if q["chunk_id"]]
    out = []
    for q in qa:
        lang = "zh" if any("一" <= c <= "鿿" for c in q["q"]) else "en"
        out.append((lang, q["q"], make_audio(tts, q["q"])))
        if len(out) >= n:
            break
    return out


async def _collect(pipe: Pipeline, seen: dict, stop: asyncio.Event) -> None:
    """Drain the out queue like a well-behaved client (otherwise TTS blocks on backpressure)."""
    while not stop.is_set():
        try:
            k, m = await asyncio.wait_for(pipe.out.get(), 0.05)
        except asyncio.TimeoutError:
            continue
        if k == "json":
            seen[m["type"]] = m


def _finish(metrics: dict | None, seen: dict) -> dict | None:
    if metrics is None:
        print("   no metrics; events seen:", {k: v for k, v in seen.items() if k in ("final", "context")})
        return None
    return {**metrics, "heard": seen.get("final", {}).get("text"), "context": seen.get("context", {}).get("ids")}


async def one_turn(models, s: Settings, lang: str, pcm: np.ndarray, tail_s: float = 1.6) -> dict | None:
    """Audio mode: real ASR on (synthetic) speech, real-time paced. Includes ASR mistakes."""
    pipe = Pipeline(models.make_asr(lang), models.retriever, models.llm, models.tts, s, models.executors)
    seen, stop = {}, asyncio.Event()
    col = asyncio.create_task(_collect(pipe, seen, stop))
    x = np.concatenate([np.zeros(SR // 2, dtype=np.int16), pcm, np.zeros(int(tail_s * SR), dtype=np.int16)])
    t_next = time.perf_counter()
    for i in range(0, len(x), FRAME):
        await pipe.on_audio(x[i:i + FRAME].tobytes())
        t_next += FRAME / SR
        await asyncio.sleep(max(0, t_next - time.perf_counter()))
    for _ in range(400):  # up to 20 s for the reply to finish
        if "metrics" in seen or (pipe._turn is None and "final" in seen and pipe._turn_active() is False and "metrics" in seen):
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.05)
    stop.set(); await col; await pipe.close()
    return _finish(seen.get("metrics"), seen)


async def oracle_turn(models, s: Settings, lang: str, text: str) -> dict | None:
    """Oracle-text mode: skip ASR (speech_end = now) to time retrieval + LLM + TTS on correct text."""
    pipe = Pipeline(models.make_asr(lang), models.retriever, models.llm, models.tts, s, models.executors)
    seen, stop = {}, asyncio.Event()
    col = asyncio.create_task(_collect(pipe, seen, stop))
    pipe._last_voice = time.perf_counter()
    await pipe._on_final(text)
    await asyncio.wait_for(pipe.wait_idle(), 30)
    await asyncio.sleep(0.1)
    stop.set(); await col; await pipe.close()
    return _finish(seen.get("metrics"), seen)


def summarize(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k in r if k != "type"})
    return {k: {"p50": round(percentile([r[k] for r in rows if k in r], 50), 1),
                "p95": round(percentile([r[k] for r in rows if k in r], 95), 1)} for k in keys}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="0 = every answerable question once")
    ap.add_argument("--speculate", choices=["on", "off", "default", "ab"], default="default", help="A/B the shadow turn")
    ap.add_argument("--audio-only", action="store_true", help="skip the oracle-text pass (A/B runs)")
    ap.add_argument("--force", action="store_true", help="run on a busy host (results marked quiet=false)")
    ap.add_argument("--out", default=str(ROOT / "results" / "bench.json"))
    a = ap.parse_args()
    from vora.hostcheck import perf_skip_reason
    reason = perf_skip_reason()
    if reason and not a.force:
        raise SystemExit(f"refusing to benchmark: {reason}. Use --force to record anyway (quiet=false).")
    from vora.server import Models
    s = Settings() if a.speculate == "default" else Settings(speculate=(a.speculate == "on"))
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    models = Models.load(s)
    qs = questions(models.tts, 10_000)
    a.n = a.n or len(qs)
    rows, orows, skipped = [], [], 0
    arms = [("off", Settings(speculate=False)), ("on", Settings(speculate=True))] if a.speculate == "ab" else [(None, s)]
    for i in range(a.n):
        lang, text, pcm = qs[i % len(qs)]
        order = arms if i % 2 == 0 else arms[::-1]     # alternate which arm runs first: drift hits both arms equally
        for arm, st in order:
            m = await one_turn(models, st, lang, pcm)
            if m is None:
                skipped += 1
                continue
            m["q"], m["lang"], m["arm"] = text, lang, arm
            rows.append(m)
            print(f"audio  {i:2d} {arm} {lang} heard={m.get('heard')!r} ctx={m.get('context')} total={m.get('total')} asr={m.get('asr_final')} llm={m.get('rag_first_token')} tts={m.get('tts_first_chunk')}")
        o = None if a.audio_only else await oracle_turn(models, s, lang, text)
        if o:
            o["q"], o["lang"] = text, lang
            orows.append(o)
            print(f"oracle {i:2d} {lang} ctx={o.get('context')} total={o.get('total')} llm={o.get('rag_first_token')} tts={o.get('tts_first_chunk')}")
    num = lambda rs: [{k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)} for r in rs]
    lat, olat = summarize(num(rows)), summarize(num(orows))
    est = {"p50": round(lat["asr_final"]["p50"] + olat["total"]["p50"], 1), "p95": round(lat["asr_final"]["p95"] + olat["total"]["p95"], 1)} if olat else {}
    from vora.hostcheck import is_quiet
    res = {"quiet": is_quiet(), "host": host(), "n": len(rows), "skipped": skipped, "latency_ms_audio_mode": lat, "latency_ms_oracle_text": olat, "estimated_total_ms": est, "speculate": a.speculate, "rss_mb_after_load": round(proc.memory_info().rss / 2**20),
           "rss_delta_load_mb": round((proc.memory_info().rss - rss0) / 2**20), "rows": rows, "oracle_rows": orows}
    p = ROOT / "results"
    p.mkdir(exist_ok=True)
    open(a.out, "w").write(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: res[k] for k in ("host", "n", "skipped", "latency_ms_audio_mode", "latency_ms_oracle_text", "estimated_total_ms", "rss_mb_after_load")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
