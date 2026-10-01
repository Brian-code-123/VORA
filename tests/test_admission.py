import asyncio
import time

import pytest

from tests.test_pipeline import FakeAsr, FakeLlm, FakeRetriever, FakeTts, drain, fin
from vora.config import Settings
from vora.llm import BUSY
from vora.pipeline import Admission, Executors, Pipeline


def mk(execs, q, llm, tts=None):
    return Pipeline(FakeAsr([[fin(q)]]), FakeRetriever(), llm, tts or FakeTts(), Settings(queue_max=64), execs)


def test_admission_counts_and_caps():
    a = Admission(2)
    assert a.try_acquire() and a.try_acquire() and not a.try_acquire()
    a.release()
    assert a.try_acquire()
    a.release(); a.release(); a.release()
    assert a.waiting == 0   # never negative


async def test_third_waiting_turn_gets_busy_immediately():
    ex = Executors(admission=Admission(2))
    slow = FakeLlm(tokens=("Okay ", "sure. "), delay=0.4)
    p1, p2, p3 = (mk(ex, f"question {i}", slow) for i in (1, 2, 3))
    for p in (p1, p2):
        await p.on_audio(b"\x00\x00" * 1600)
    await asyncio.sleep(0.1)
    t0 = time.perf_counter()
    await p3.on_audio(b"\x00\x00" * 1600)
    for _ in range(60):
        if p3.tts.said:
            break
        await asyncio.sleep(0.01)
    assert time.perf_counter() - t0 < 0.5
    assert " ".join(p3.tts.said).startswith(BUSY["en"].split(",")[0])
    assert "question 3" not in slow.calls
    for p in (p1, p2, p3):
        await p.wait_idle()


async def test_slot_released_on_cancel_error_and_normal_end():
    ex = Executors(admission=Admission(2))

    class Boom(FakeLlm):
        def stream(self, q, hits, cancel=None):
            raise RuntimeError("boom")
            yield

    ok = mk(ex, "fine question", FakeLlm(delay=0.01))
    await ok.on_audio(b"\x00\x00" * 1600)
    await ok.wait_idle()
    bad = mk(ex, "bad question", Boom())
    await bad.on_audio(b"\x00\x00" * 1600)
    await bad.wait_idle()
    slow = mk(ex, "cancelled question", FakeLlm(delay=0.2))
    await slow.on_audio(b"\x00\x00" * 1600)
    await asyncio.sleep(0.05)
    await slow.close()
    await asyncio.sleep(0.6)
    assert ex.admission.waiting == 0


async def test_cancel_before_worker_starts_does_not_double_release():
    ex = Executors(admission=Admission(2))
    # occupy both LLM executor threads so the third turn's worker stays queued
    from concurrent.futures import ThreadPoolExecutor
    ex.llm = ThreadPoolExecutor(1)
    blocker = mk(ex, "blocker", FakeLlm(delay=0.3, tokens=("a ", "b ")))
    await blocker.on_audio(b"\x00\x00" * 1600)
    queued = mk(ex, "queued question", FakeLlm(delay=0.01))
    await queued.on_audio(b"\x00\x00" * 1600)
    await asyncio.sleep(0.1)
    await queued.close()               # cancelled while its worker has not started
    await blocker.wait_idle()
    await asyncio.sleep(0.5)
    assert ex.admission.waiting == 0
