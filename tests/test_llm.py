import threading
import time

import pytest

from vora.config import Settings
from vora.llm import BUSY, UNSURE, Llm
from vora.rag.store import Hit

S = Settings()
pytestmark = pytest.mark.skipif(not (S.models_dir / "llm").exists(), reason="models not fetched")

WARRANTY_EN = Hit("E03", "Warranty: the VORA-X200 warranty period is 2 years. The VORA-M100 warranty period is 1 year.", 0.9)
WARRANTY_ZH = Hit("Z03", "保修：VORA-X200 的保修期是 2 年，VORA-M100 的保修期是 1 年。", 0.9)


@pytest.fixture(scope="module")
def llm():
    return Llm(S)


def fake_stream(text):
    def _f(**kw):
        for ch in text:
            yield {"choices": [{"delta": {"content": ch}}]}
    return _f


def test_streams_multiple_tokens_incrementally(llm):
    toks = list(llm.stream("How long is the warranty on the VORA-X200?", [WARRANTY_EN]))
    assert len(toks) >= 3
    assert "2" in "".join(toks) or "two" in "".join(toks).lower()


def test_answer_uses_context_term(llm):
    out = "".join(llm.stream("How long is the VORA-X200 warranty?", [WARRANTY_EN])).lower()
    assert "year" in out or "2" in out


def test_zh_answer(llm):
    out = "".join(llm.stream("X200保修期多久", [WARRANTY_ZH]))
    assert "2" in out or "两" in out or "二" in out


def test_cancel_stops_within_one_token(llm):
    it = llm.stream("Explain the VORA-X200 warranty in detail.", [WARRANTY_EN])
    n = 0
    for _ in it:
        n += 1
        if n == 2:
            llm.cancel()
    assert n <= 3
    assert llm._lock.acquire(timeout=1)  # lock released after cancel
    llm._lock.release()


def test_empty_hits_no_model_call(llm, monkeypatch):
    def boom(**kw):
        raise AssertionError("model called")
    monkeypatch.setattr(llm.llm, "create_chat_completion", boom)
    assert list(llm.stream("what is the weather", [])) == [UNSURE["en"]]
    assert list(llm.stream("今天天气", [])) == [UNSURE["zh"]]


def test_extractive_fallback_when_no_keyword_overlap(llm, monkeypatch):
    monkeypatch.setattr(llm.llm, "create_chat_completion", fake_stream("Bananas are yellow fruits that grow in tropical places."))
    out = "".join(llm.stream("How long is the VORA-X200 warranty?", [WARRANTY_EN]))
    assert out == "Warranty: the VORA-X200 warranty period is 2 years."


def test_lock_timeout_returns_busy(llm):
    llm._lock.acquire()
    try:
        t0 = time.time()
        assert list(llm.stream("hi", [WARRANTY_EN])) == [BUSY["en"]]
        assert 2.5 < time.time() - t0 < 5
    finally:
        llm._lock.release()


def test_two_users_alternating_keeps_prefix_cache(llm):
    other = Hit("E05", "LED colors: blue means the box is listening, green means it is speaking.", 0.9)
    list(llm.stream("warranty of x200?", [WARRANTY_EN]))
    list(llm.stream("what does blue mean?", [other]))
    list(llm.stream("warranty of x200?", [WARRANTY_EN]))
    assert len(llm.llm.cache.cache_state) >= 2


@pytest.mark.perf
def test_cold_retrieval_plus_first_token_under_500ms(llm):
    from vora.rag.retriever import Retriever
    r = Retriever(S)
    q = "How long is the warranty on the VORA-X200?"
    list(llm.stream(q, r.search(q)))  # warm system prefix
    q2 = "what does the red light mean"
    t0 = time.perf_counter()
    it = llm.stream(q2, r.search(q2))
    next(it)
    dt = (time.perf_counter() - t0) * 1000
    list(it)
    assert dt < 500, dt


@pytest.mark.perf
def test_warm_first_token_under_300ms(llm):
    q = "How long is the warranty on the VORA-X200?"
    list(llm.stream(q, [WARRANTY_EN]))
    t0 = time.perf_counter()
    it = llm.stream("How long is the VORA-X200 warranty period?", [WARRANTY_EN])
    next(it)
    dt = (time.perf_counter() - t0) * 1000
    list(it)
    assert dt < 300, dt
