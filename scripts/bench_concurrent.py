"""Two users at once on the real models (shared LLM/TTS/executors). Usage: bench_concurrent.py [--n 8] [--out results/concurrent.json]
Each question is spoken by user A and (a different question) by user B at the same time, real-time paced."""
import argparse
import asyncio
import json
import statistics

from scripts.bench import answerable_questions, make_audio, one_turn
from vora.config import ROOT, Settings
from vora.hostcheck import QuietMonitor, uss_mb


async def run(models, s: Settings, n: int) -> dict:
    mon = QuietMonitor()
    mon.start()
    active = [0]                    # what the server's active_sessions() would say: speculation switches off with 2 users

    async def turn(lang, pcm):
        active[0] += 1
        try:
            return await one_turn(models, s, lang, pcm, active_sessions=lambda: active[0])
        finally:
            active[0] -= 1

    qs = [(l, t) for l, t in answerable_questions() if l == "en"]
    audio = {t: make_audio(models.tts, t) for _, t in qs}
    single, pair_a, pair_b = [], [], []
    for i in range(n):
        ta, tb = qs[i % len(qs)][1], qs[(i + 7) % len(qs)][1]
        m = await turn("en", audio[ta])
        if m and m.get("context") and "total" in m:
            single.append(m["total"])
    u0 = uss_mb()
    for i in range(n):
        ta, tb = qs[i % len(qs)][1], qs[(i + 7) % len(qs)][1]
        a, b = await asyncio.gather(turn("en", audio[ta]), turn("en", audio[tb]))
        if a and a.get("context") and "total" in a:
            pair_a.append(a["total"])
        if b and b.get("context") and "total" in b:
            pair_b.append(b["total"])
    s50, a50, b50 = statistics.median(single), statistics.median(pair_a), statistics.median(pair_b)
    qm = mon.stop()
    worst = max(a50, b50)
    return {"single_p50_ms": round(s50), "user_a_p50_ms": round(a50), "user_b_p50_ms": round(b50), "concurrent_p50_ms": round(worst),
            "ratio": round(worst / s50, 2), "n_single": len(single), "n_a": len(pair_a), "n_b": len(pair_b), "n_concurrent": len(pair_a) + len(pair_b),
            "uss_delta_mb": round(uss_mb() - u0, 1), "quiet": qm["quiet"], "max_other_cpu_pct": qm["max_other_cpu_pct"],
            "speculation_during_pairs": "off (active_sessions=2)",
            "summary": f"2 users: A p50 {a50:.0f}, B p50 {b50:.0f} ms vs single {s50:.0f} ms (worst x{worst / s50:.2f})", "ok": worst <= 2 * s50}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "concurrent.json"))
    a = ap.parse_args()
    from vora.server import Models
    s = Settings()
    res = asyncio.run(run(Models.load(s), s, a.n))
    (ROOT / "results").mkdir(exist_ok=True)
    open(a.out, "w").write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
