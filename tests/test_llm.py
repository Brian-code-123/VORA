import threading
import time

import pytest

from vora.config import Settings
from vora.llm import BUSY, SYSTEM, UNSURE, Llm
from vora.rag.store import Hit

S = Settings()
pytestmark = pytest.mark.skipif(not (S.models_dir / "llm").exists(), reason="models not fetched")

WARRANTY_EN = Hit("E03", "Warranty: the VORA-X200 warranty period is 2 years. The VORA-M100 warranty period is 1 year.", 0.9)
WARRANTY_ZH = Hit("Z03", "保修：VORA-X200 的保修期是 2 年，VORA-M100 的保修期是 1 年。", 0.9)


@pytest.fixture(scope="module")
def llm():
    return Llm(S)


def fake_words(text):
    """Word-level tokens like a real model (a 6-token head must be able to hold a whole refusal phrase)."""
    import re
    toks = re.findall(r"\s*\S+", text)

    def _f(**kw):
        for t in toks:
            yield {"choices": [{"delta": {"content": t}}]}
    return _f


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


def test_alternating_users_reuse_system_prefix(llm):
    """Users alternate with different questions: the KV cache must still start with the shared system prompt."""
    other = Hit("E05", "LED colors: blue means the box is listening, green means it is speaking.", 0.9)
    list(llm.stream("warranty of x200?", [WARRANTY_EN]))
    list(llm.stream("what does blue mean?", [other]))
    sys_ids = llm.llm.tokenize(("<|im_start|>system\n" + SYSTEM + "<|im_end|>").encode(), add_bos=False, special=True)
    assert list(llm.llm._input_ids[: len(sys_ids)]) == sys_ids


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


CANT = Hit("E10", "Languages: VORA Box understands and speaks Mandarin Chinese and English. Cantonese speech is not supported.", 0.9)


def test_guard_replaces_yes_to_a_negated_fact(llm, monkeypatch):
    monkeypatch.setattr(llm.llm, "create_chat_completion", fake_words("Yes, the VORA Box understands Cantonese very well."))
    assert "".join(llm.stream("does it understand cantonese", [CANT])) == "Cantonese speech is not supported."


def test_refusal_is_replaced_by_best_sentence(llm, monkeypatch):
    from vora.guard import best_sentence
    monkeypatch.setattr(llm.llm, "create_chat_completion",
                        fake_words("The context does not provide information about the warranty period at all."))
    out = "".join(llm.stream("how long is the warranty", [WARRANTY_EN]))
    assert out == best_sentence(WARRANTY_EN.text, "how long is the warranty")


def test_decide_buffer_is_6_tokens():
    import vora.llm as m
    assert m._DECIDE_TOKENS == 6


def test_system_prompt_le_40_tokens(llm):
    assert len(llm.llm.tokenize(SYSTEM.encode())) <= 40


def test_context_trim_keeps_the_answer_sentence(llm):
    filler = " ".join(f"Sentence number {i} talks about unrelated hardware details." for i in range(30))
    chunk = Hit("X1", f"{filler} The warranty period is exactly two years. {filler}", 0.9)
    msgs = llm._messages("what is the warranty period", [chunk])
    ctx = msgs[1]["content"]
    assert "warranty period is exactly two years" in ctx
    assert len(llm.llm.tokenize(ctx.encode())) <= llm.s.max_ctx_tokens + 40


@pytest.mark.skipif(not (S.index_dir / "faiss.index").exists(), reason="needs index")
def test_prompt_tokens_le_170_for_every_eval_question(llm):
    from vora.config import ROOT
    from vora.rag.ingest import load_qa
    from vora.rag.retriever import Retriever
    r = Retriever(S)
    worst = 0
    for q in [x["q"] for x in load_qa(ROOT / "eval" / "rag_qa.jsonl") if x["chunk_id"]]:
        worst = max(worst, llm.prompt_tokens(q, r.search(q)))
    assert worst <= 170, worst


def test_polarity_checked_against_every_hit(llm, monkeypatch):
    other = Hit("E01", "VORA-X200 hardware: a 4-core ARM CPU and 4 GB RAM.", 0.95)
    monkeypatch.setattr(llm.llm, "create_chat_completion", fake_words("Yes, the VORA Box understands Cantonese very well."))
    out = "".join(llm.stream("does it understand cantonese", [other, CANT]))   # the negating chunk is NOT the top hit
    assert out == "Cantonese speech is not supported."


def test_invented_number_replaced_by_chunk_sentence(llm, monkeypatch):
    price = Hit("E12", "Price and returns: the VORA-X200 costs 199 USD. You can return either model within 30 days.", 0.9)
    monkeypatch.setattr(llm.llm, "create_chat_completion", fake_words("The X200 is priced at $1,000 in the store."))
    out = "".join(llm.stream("how much does the x200 cost", [price]))
    assert "199" in out and "1,000" not in out


def test_yes_no_question_answered_extractively_without_the_model(llm, monkeypatch):
    def boom(**kw):
        raise AssertionError("model must not be called for yes/no questions")
    monkeypatch.setattr(llm.llm, "create_chat_completion", boom)
    assert "".join(llm.stream("will a factory reset delete the firmware", [Hit("E08", "Factory reset: hold the reset button for 10 seconds. This erases all settings and imported documents. Firmware is kept.", 0.9)])) \
        .lower().startswith(("factory reset", "firmware is kept", "this erases")) is True


def test_quantity_question_without_a_figure_gets_the_sentence_appended(llm, monkeypatch):
    mic = Hit("E16", "Microphones: the X200 has a 4-microphone array with a pickup range of 5 meters. The M100 has 2 microphones with a range of 3 meters.", 0.9)
    monkeypatch.setattr(llm.llm, "create_chat_completion", fake_words("The M100 can hear you from quite a distance away."))
    out = "".join(llm.stream("how far can the m100 hear me", [mic]))
    assert "3 meters" in out and out.startswith("The M100 can hear you")
