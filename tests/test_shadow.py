import asyncio
import time

import pytest

from tests.test_pipeline import FakeAsr, FakeLlm, FakeRetriever, FakeTts, drain, fin, part
from vora.config import Settings
from vora.pipeline import Admission, Executors, Pipeline

Q = "how long is the warranty"


_PIPES: list = []


@pytest.fixture(autouse=True)
async def _close_pipes():
    yield
    for p in _PIPES:
        await p.close()      # a failed assertion must not leave blocked TTS threads (they hang interpreter exit)
    _PIPES.clear()


def peek(p):
    """Items waiting in the out queue without draining it."""
    return list(p.out._queue)


def mk(script, llm=None, tts=None, active=lambda: 1, execs=None, **st):
    st.setdefault("speculate", True)
    st.setdefault("speculate_min_cores", 1)
    s = Settings(queue_max=64, **st)
    p = Pipeline(FakeAsr(script), FakeRetriever(), llm or FakeLlm(delay=0.01), tts or FakeTts(chunks=3, delay=0.005), s,
                 execs or Executors(), active_sessions=active)
    _PIPES.append(p)
    return p


async def feed(p, n=1):
    for _ in range(n):
        await p.on_audio(b"\x00\x00" * 1600)


def audio_count(p):
    return sum(1 for m in drain(p) if m[0] == "audio")


async def test_promoted_first_audio_within_50ms_of_final():
    p = mk([[part(Q, True)], [fin(Q)]])
    await feed(p)
    await asyncio.sleep(0.6)            # shadow retrieves, generates and pre-renders while the user is silent
    assert p._shadow is not None
    assert [m for m in peek(p) if m[0] == "audio" or m[1]["type"] in ("token", "context", "final", "metrics")] == []   # nothing leaked yet
    t0 = time.perf_counter()
    await feed(p)                        # the final arrives
    assert any(m[0] == "audio" for m in drain(p)), "pre-rendered audio must be released at once"
    assert time.perf_counter() - t0 < 0.05 + 0.01
    await p.wait_idle()
    await p.close()


async def test_promoted_turn_reports_speculated_and_final_text():
    p = mk([[part(Q, True)], [fin(Q)]])
    await feed(p)
    await asyncio.sleep(0.5)
    await feed(p)
    await p.wait_idle()
    msgs = [m[1] for m in drain(p) if m[0] == "json"]
    types = [m["type"] for m in msgs]
    assert types.index("final") < types.index("context") < types.index("metrics")
    m = [x for x in msgs if x["type"] == "metrics"][0]
    assert m["speculated"] is True and p.llm.calls == [Q]      # one generation, not two


async def test_cancelled_when_final_differs_no_audio_leaked():
    p = mk([[part(Q, True)], [fin("what is the wake word")]], llm=FakeLlm(delay=0.2, tokens=("Hello ", "there. ")))
    await feed(p)
    await asyncio.sleep(0.3)
    await feed(p)
    assert audio_count(p) == 0                                  # the shadow's pre-rendered audio never reached the client
    await p.wait_idle()
    assert p.llm.calls == [Q, "what is the wake word"]
    await p.close()


async def test_cancelled_when_user_keeps_talking():
    p = mk([[part(Q, True)], [part("how long is the warranty on the x200", False)]])
    await feed(p)
    await asyncio.sleep(0.2)
    assert p._shadow is not None
    await feed(p)
    assert p._shadow is None and audio_count(p) == 0


async def test_not_started_for_short_text():
    p = mk([[part("hi there", True)]])
    await feed(p)
    await asyncio.sleep(0.1)
    assert p._shadow is None and p.llm.calls == []


async def test_not_started_when_other_session_active():
    p = mk([[part(Q, True)]], active=lambda: 2)
    await feed(p)
    await asyncio.sleep(0.1)
    assert p._shadow is None and p.llm.calls == []


async def test_not_started_when_cores_below_min():
    p = mk([[part(Q, True)]], speculate_min_cores=10_000)
    await feed(p)
    await asyncio.sleep(0.1)
    assert p._shadow is None and p.llm.calls == []


async def test_speculate_off_setting():
    p = mk([[part(Q, True)]], speculate=False)
    await feed(p)
    await asyncio.sleep(0.1)
    assert p._shadow is None and p.llm.calls == []


async def test_buffer_bounded_to_two_sentences_before_promotion():
    llm = FakeLlm(tokens=("One. ", "Two. ", "Three. ", "Four. ", "Five. "), delay=0.01)
    p = mk([[part(Q, True)]], llm=llm)
    await feed(p)
    await asyncio.sleep(0.8)
    assert len(p.tts.said) <= 2 and not [m for m in peek(p) if m[0] == "audio"]
    await p.close()


async def test_shadow_text_not_in_spoken_until_promoted():
    p = mk([[part(Q, True)], [fin(Q)]])
    await feed(p)
    await asyncio.sleep(0.4)
    assert len(p._spoken) == 0
    await feed(p)
    assert len(p._spoken) > 0
    await p.wait_idle()
    await p.close()


async def test_cancel_releases_admission_and_llm():
    ex = Executors(admission=Admission(2))
    p = mk([[part(Q, True)], [fin("something else entirely")]], llm=FakeLlm(delay=0.3, tokens=("a ", "b ", "c ")), execs=ex)
    await feed(p)
    await asyncio.sleep(0.2)
    await feed(p)
    await p.wait_idle()
    await asyncio.sleep(0.4)
    assert ex.admission.waiting == 0
    await p.close()


async def test_shadow_cancelled_when_second_session_joins():
    n = {"v": 1}
    p = mk([[part(Q, True)], []], active=lambda: n["v"])
    await feed(p)
    await asyncio.sleep(0.2)
    assert p._shadow is not None
    n["v"] = 2
    await feed(p)
    assert p._shadow is None


async def test_shadow_gives_up_when_admission_busy_and_never_speaks_busy():
    ex = Executors(admission=Admission(1))
    ex.admission.try_acquire()
    p = mk([[part(Q, True)]], execs=ex)
    await feed(p)
    await asyncio.sleep(0.3)
    assert p.llm.calls == [] and p.tts.said == []
    await p.close()


async def test_close_cancels_shadow():
    p = mk([[part(Q, True)]])
    await feed(p)
    await asyncio.sleep(0.2)
    await p.close()
    assert p._shadow is None
