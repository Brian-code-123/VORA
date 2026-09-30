import asyncio
import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest

from vora.asr import AsrEvent
from vora.config import Settings
from vora.llm import UNSURE, Llm
from vora.pipeline import Executors, Pipeline
from vora.rag.store import Hit

HIT = Hit("E03", "Warranty is 2 years.", 0.9)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeAsr:
    """script: list of event-lists, one per feed() call; blocking sleep simulates decode."""
    def __init__(self, script, delay=0.0, clock=None, step=0.1):
        self.script, self.i, self.delay, self.clock, self.step = script, 0, delay, clock, step

    def feed(self, pcm):
        time.sleep(self.delay)
        if self.clock is not None:
            self.clock.t += self.step
        ev = self.script[self.i] if self.i < len(self.script) else []
        self.i += 1
        return ev


class FakeRetriever:
    def __init__(self, hits=(HIT,), delay=0.0):
        self.calls, self.hits, self.delay = [], list(hits), delay

    def search(self, q):
        self.calls.append(q)
        time.sleep(self.delay)
        return self.hits


class FakeLlm:
    def __init__(self, tokens=("Warranty ", "is ", "two ", "years. ", "Bye."), delay=0.0):
        self.tokens, self.delay, self.calls, self.cancelled = tokens, delay, [], 0

    def stream(self, q, hits, cancel=None):
        self.calls.append(q)
        done = False
        try:
            for t in self.tokens:
                time.sleep(self.delay)
                yield t
            done = True
        finally:
            if not done:  # consumer closed the stream early (turn cancelled)
                self.cancelled += 1


class FakeTts:
    def __init__(self, chunks=3, delay=0.0):
        self.chunks, self.delay, self.said, self.cancelled = chunks, delay, [], 0

    def synth(self, text, cancel=None):
        self.said.append(text)
        done = False
        try:
            for _ in range(self.chunks):
                time.sleep(self.delay)
                yield b"\x01\x00" * 100
            done = True
        finally:
            if not done:  # consumer closed the stream early (turn cancelled / disconnect)
                self.cancelled += 1


def P(script, retriever=None, llm=None, tts=None, filler=False, queue_max=64, delay=0.0, clock=time.perf_counter, **st):
    s = Settings(filler=filler, queue_max=queue_max, **st)
    asr = FakeAsr(script, delay, clock=clock if isinstance(clock, Clock) else None)
    return Pipeline(asr, retriever or FakeRetriever(), llm or FakeLlm(), tts or FakeTts(), s, Executors(), clock)


def drain(p):
    out = []
    while not p.out.empty():
        out.append(p.out.get_nowait())
    return out


def part(t, stable=False):
    return AsrEvent("partial", t, stable)


def fin(t):
    return AsrEvent("final", t, True)


async def feed_all(p, n):
    for _ in range(n):
        await p.on_audio(b"\x00\x00" * 1600)


async def test_empty_final_no_llm_call():
    p = P([[fin("  ")]])
    await feed_all(p, 1)
    await p.wait_idle()
    assert p.llm.calls == []


async def test_prefetch_reused_when_final_matches_partial():
    p = P([[part("how long is the warranty", True)], [fin("how long is the warranty")]])
    await feed_all(p, 2)
    await p.wait_idle()
    assert p.retriever.calls == ["how long is the warranty"]


async def test_prefetch_rate_limited():
    p = P([[part("how long is", True)], [part("how long is the warranty", True)], [part("how long is the warranty period", True)]],
          clock=lambda: 100.0)
    await feed_all(p, 3)
    await asyncio.sleep(0.05)
    assert len(p.retriever.calls) == 1


async def test_mixed_zh_en_question_forwarded_verbatim():
    q = "請問 VORA Box 支援 WiFi 嗎"
    p = P([[fin(q)]])
    await feed_all(p, 1)
    await p.wait_idle()
    assert p.llm.calls == [q] and p.retriever.calls == [q]


async def test_no_hits_speaks_fallback():
    llm = Llm.__new__(Llm)  # empty-hits path never touches the model
    p = P([[fin("what is the weather")]], retriever=FakeRetriever([]), llm=llm)
    await feed_all(p, 1)
    await p.wait_idle()
    assert " ".join(p.tts.said) == UNSURE["en"]  # chunker splits it into two sentences


async def test_total_latency_recorded():
    p = P([[part("hello there", False)], [fin("hello there")]])
    await feed_all(p, 2)
    await p.wait_idle()
    m = [x[1] for x in drain(p) if x[0] == "json" and x[1]["type"] == "metrics"][0]
    assert m["total"] > 0 and m["asr_final"] >= 0 and m["first_content_audio_ms"] == m["ttfa_ms"]


async def test_filler_off_by_default_and_reported_separately():
    p = P([[fin("hello there")]], tts=FakeTts(delay=0.02))
    await feed_all(p, 1)
    await p.wait_idle()
    assert not any(t in ("Sure, ", "好的，") for t in p.tts.said)
    p2 = P([[fin("hello there")]], llm=FakeLlm(delay=0.05), tts=FakeTts(delay=0.01), filler=True)
    await feed_all(p2, 1)
    await p2.wait_idle()
    m = [x[1] for x in drain(p2) if x[0] == "json" and x[1]["type"] == "metrics"][0]
    assert p2.tts.said[0] == "Sure, "
    assert m["ttfa_ms"] < m["first_content_audio_ms"]


async def test_barge_in_cancels_and_drains():
    p = P([[fin("hello there")], [], [part("wait", False)], [part("wait stop", False)]],
          llm=FakeLlm(delay=0.01), tts=FakeTts(chunks=200, delay=0.01), queue_max=4)
    await feed_all(p, 2)
    await asyncio.sleep(0.15)  # reply audio is flowing / queue full
    await feed_all(p, 2)
    await asyncio.sleep(0.1)
    assert p.tts.cancelled >= 1  # worker observed this turn's cancel event
    msgs = drain(p)
    idx = max(i for i, m in enumerate(msgs) if m[0] == "json" and m[1]["type"] == "cancel")
    assert not any(m[0] == "audio" for m in msgs[idx + 1:])
    await asyncio.sleep(0.1)
    assert not any(m[0] == "audio" for m in drain(p))  # nothing stale after cancel


async def test_single_chunk_blip_does_not_barge_in():
    p = P([[fin("hello there")], [], [part("uh", False)], [], []], llm=FakeLlm(delay=0.02))
    await feed_all(p, 5)
    await p.wait_idle()
    assert p.llm.cancelled == 0
    assert any(m[0] == "json" and m[1]["type"] == "metrics" for m in drain(p))


async def test_new_final_cancels_previous_turn():
    p = P([[fin("first question")], [fin("second question")]], llm=FakeLlm(delay=0.05), tts=FakeTts(delay=0.02))
    await feed_all(p, 1)
    await asyncio.sleep(0.08)
    await feed_all(p, 1)
    await p.wait_idle()
    await asyncio.sleep(0.15)
    assert p.llm.cancelled >= 1 and p.llm.calls == ["first question", "second question"]
    types = [m[1]["type"] for m in drain(p) if m[0] == "json"]
    assert types.count("metrics") == 1 and "cancel" in types


async def test_event_loop_lag_under_50ms_during_decode():
    p = P([[part("a", False)]] * 6, delay=0.12)
    lag = []

    async def ticker():
        while True:
            t = time.perf_counter()
            await asyncio.sleep(0.01)
            lag.append(time.perf_counter() - t - 0.01)

    tk = asyncio.create_task(ticker())
    await feed_all(p, 4)
    tk.cancel()
    assert max(lag) < 0.05, max(lag)


LOUD = (np.ones(1600) * 4000).astype(np.int16).tobytes()
QUIET = b"\x00\x00" * 1600


async def test_barge_in_with_real_asr_partial_pattern():
    """Real ASR at 100 ms frames changes its text about every 3rd feed (C . s C . s ...): must still barge in."""
    clk = Clock()
    p = P([[fin("hello there")], [], [], [part("wait", False)], [part("wait", True)], [], [part("wait stop", False)]],
          llm=FakeLlm(delay=0.05), tts=FakeTts(chunks=300, delay=0.01), clock=clk)
    await feed_all(p, 3)
    await asyncio.sleep(0.15)
    await feed_all(p, 4)
    await asyncio.sleep(0.1)
    types = [m[1]["type"] for m in drain(p) if m[0] == "json"]
    assert "cancel" in types


async def test_speech_end_is_last_voiced_frame():
    clk = Clock()
    p = P([[], [], [], [fin("hello there")]], clock=clk)
    for frame in (QUIET, LOUD, QUIET, QUIET):
        await p.on_audio(frame)
    await p.wait_idle()
    m = [x[1] for x in drain(p) if x[0] == "json" and x[1]["type"] == "metrics"][0]
    assert m["asr_final"] == pytest.approx(300.0, abs=1)   # last loud frame at t=0.1, final at t=0.4


async def test_cancelled_turn_does_not_kill_next_turn_via_shared_prefetch():
    p = P([[part("how long is the warranty", True)], [fin("how long is the warranty")], [fin("how long is the warranty")]],
          retriever=FakeRetriever(delay=0.2))
    await feed_all(p, 2)
    await asyncio.sleep(0.05)   # turn 1 is waiting on the prefetch
    await feed_all(p, 1)        # same question again: cancels turn 1, turn 2 must still finish
    await p.wait_idle()
    types = [m[1]["type"] for m in drain(p) if m[0] == "json"]
    assert "metrics" in types and len(p.llm.calls) == 1


async def test_prefetch_not_reused_when_final_extends_partial():
    p = P([[part("how much does the", True)], [fin("how much does the m100 cost")]])
    await feed_all(p, 2)
    await p.wait_idle()
    assert p.retriever.calls == ["how much does the", "how much does the m100 cost"]


async def test_echo_of_own_reply_is_ignored():
    p = P([[fin("first question")], [fin("warranty is two years")]], llm=FakeLlm(tokens=("Warranty ", "is ", "two ", "years. ")))
    await feed_all(p, 1)
    await p.wait_idle()
    await feed_all(p, 1)   # the mic heard the speaker
    await p.wait_idle()
    assert p.llm.calls == ["first question"]


async def test_stalled_client_raises():
    from vora.pipeline import ClientStalled
    p = P([[fin("q one")], [fin("q two")], [fin("q three")]], queue_max=1, stall_timeout_s=0.2,
          tts=FakeTts(chunks=50, delay=0.001))
    await feed_all(p, 1)
    await asyncio.sleep(0.6)   # nobody reads p.out
    with pytest.raises(ClientStalled):
        await feed_all(p, 1)


@pytest.mark.skipif(not (Settings().models_dir / "asr_en").exists(), reason="models not fetched")
async def test_barge_in_with_real_asr_on_real_speech():
    """Review finding: with the real recognizer at 100 ms frames, talking over the reply must cancel it."""
    import wave

    from vora.asr import AsrSession, load_recognizers
    s = Settings()
    recs = load_recognizers(s)
    p = Pipeline(AsrSession(recs, "en"), FakeRetriever(), FakeLlm(delay=0.05), FakeTts(chunks=400, delay=0.01), s, Executors())
    await p._on_final("earlier question")          # a reply is playing
    await asyncio.sleep(0.2)
    with wave.open(str(s.models_dir / "asr_en/test_wavs/1.wav")) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    types = []
    for i in range(0, len(x), 1600):
        await p.on_audio(x[i:i + 1600].tobytes())
        types += [m[1]["type"] for m in drain(p) if m[0] == "json"]
        if "cancel" in types:
            break
    await p.close()
    assert "cancel" in types, "real speech over the reply never interrupted it"
