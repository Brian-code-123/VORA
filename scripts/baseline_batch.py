"""Batch (non-streaming) baseline on the same models and questions: wait for the whole utterance -> retrieve ->
generate the FULL answer -> synthesise the FULL answer -> only then play. Compared with the streaming pipeline.
Endpoint wait is taken from the streaming bench (both systems must wait for the user to stop)."""
import json
import time

from scripts.bench import host, percentile, questions
from vora.config import ROOT, Settings
from vora.server import Models


def main(n: int = 30) -> None:
    s = Settings()
    m = Models.load(s)
    qs = questions(m.tts, 12)
    stream = json.loads((ROOT / "results" / "bench.json").read_text())
    endpoint = stream["latency_ms_audio_mode"]["asr_final"]
    rows = []
    for i in range(n):
        lang, text, _ = qs[i % len(qs)]
        t0 = time.perf_counter()
        hits = m.retriever.search(text)
        t1 = time.perf_counter()
        answer = "".join(m.llm.stream(text, hits))
        t2 = time.perf_counter()
        audio = b"".join(m.tts.synth(answer))
        t3 = time.perf_counter()
        rows.append({"q": text, "retrieve_ms": (t1 - t0) * 1e3, "llm_full_ms": (t2 - t1) * 1e3, "tts_full_ms": (t3 - t2) * 1e3,
                     "after_endpoint_ms": (t3 - t0) * 1e3, "answer_chars": len(answer), "audio_bytes": len(audio)})
        print(f"{i:2d} after_endpoint={rows[-1]['after_endpoint_ms']:.0f} ms | {text}")
    a = [r["after_endpoint_ms"] for r in rows]
    so = stream["latency_ms_oracle_text"]["total"]
    res = {"host": host(), "n": n, "endpoint_ms": endpoint,
           "batch_after_endpoint_ms": {"p50": round(percentile(a, 50), 1), "p95": round(percentile(a, 95), 1)},
           "streaming_after_endpoint_ms": so,
           "batch_total_ms": {k: round(endpoint[k] + v, 1) for k, v in (("p50", percentile(a, 50)), ("p95", percentile(a, 95)))},
           "streaming_total_estimated_ms": stream["estimated_total_ms"], "rows": rows}
    (ROOT / "results" / "baseline.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print({k: res[k] for k in ("batch_after_endpoint_ms", "streaming_after_endpoint_ms", "batch_total_ms", "streaming_total_estimated_ms")})


if __name__ == "__main__":
    main()
