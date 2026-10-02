import numpy as np
import pytest

from vora.asr import AsrSession
from vora.endpoint import looks_incomplete, strip_disfluency

SR = 16000


@pytest.mark.parametrize("text,lang,expected", [
    ("how long is the", "en", True), ("what about the m100 and", "en", True), ("how long is the warranty", "en", False),
    ("", "en", False), ("what is", "en", True), ("does it support wifi", "en", False),
    ("保修期是", "zh", True), ("因为", "zh", True), ("保修期是几年", "zh", False), ("请问x200的", "zh", True), ("", "zh", False),
])
def test_looks_incomplete(text, lang, expected):
    assert looks_incomplete(text, lang) is expected


@pytest.mark.parametrize("text,lang,expected", [
    ("um how long is the warranty uh", "en", "how long is the warranty"), ("uh", "en", ""),
    ("the umbrella is red", "en", "the umbrella is red"), ("嗯保修期多久呃", "zh", "保修期多久"), ("嗯", "zh", ""), ("退款金额是多少", "zh", "退款金额是多少"), ("额定功率是多少", "zh", "额定功率是多少"),
])
def test_strip_disfluency(text, lang, expected):
    assert strip_disfluency(text, lang) == expected


class FakeRec:
    """Scripted recognizer. schedule = [(t_sec, text, endpoint)]: state at audio time t is the last entry with entry.t <= t.
    t counts audio fed AFTER the 0.8 s pre-roll."""

    def __init__(self, schedule):
        self.schedule, self.fed, self.reset_calls = schedule, 0, 0
        self.PRE = int(0.8 * SR)

    def create_stream(self):
        self.fed = 0
        return self

    def accept_waveform(self, sr, x):
        self.fed += len(x)

    def is_ready(self, s):
        return False

    def decode_stream(self, s): ...

    def _state(self):
        t = max(0.0, (self.fed - self.PRE) / SR)
        cur = ("", False)
        for ts, text, ep in self.schedule:
            if ts <= t + 1e-9:
                cur = (text, ep)
        return cur

    def get_result(self, s):
        return self._state()[0]

    def is_endpoint(self, s):
        return self._state()[1]

    def reset(self, s):
        self.reset_calls += 1


def run(schedule, seconds, hold_ms=500, cap_ms=1200, lang="en"):
    rec = FakeRec(schedule)
    sess = AsrSession({"en": rec, "zh": rec}, lang, hold_ms=hold_ms, hold_total_cap_ms=cap_ms)
    out = []   # (t, event)
    for i in range(int(seconds * 10)):
        for e in sess.feed(np.zeros(1600, dtype=np.int16).tobytes()):
            out.append(((i + 1) / 10, e))
    return rec, out


def finals(out):
    return [(t, e) for t, e in out if e.kind == "final"]


def test_incomplete_endpoint_no_final_no_reset_until_hold_expires():
    rec, out = run([(1.0, "how long is the", False), (1.5, "how long is the", True)], 3.0)
    f = finals(out)
    assert len(f) == 1 and f[0][1].text == "how long is the"
    assert f[0][0] >= 1.5 + 0.5 - 0.1            # held ~500 ms of audio before releasing
    assert rec.reset_calls == 0
    assert f[0][1].held_ms >= 400


def test_continuation_extends_same_utterance():
    rec, out = run([(1.0, "how long is the", False), (1.5, "how long is the", True),
                    (1.7, "how long is the warranty", False), (2.2, "how long is the warranty", True)], 3.0)
    f = finals(out)
    assert [e.text for _, e in f] == ["how long is the warranty"]
    assert rec.reset_calls == 0


def test_complete_sentence_final_not_delayed():
    _, out = run([(1.0, "how long is the warranty", False), (1.5, "how long is the warranty", True)], 3.0)
    f = finals(out)
    assert len(f) == 1 and f[0][0] <= 1.6 and f[0][1].held_ms == 0


def test_rambling_capped_at_total_cap():
    sched, txt, t = [], "and", 1.0
    for _ in range(12):
        sched.append((t, txt, False)); sched.append((t + 0.3, txt, True))
        txt += " and"; t += 0.6
    _, out = run(sched, 8.0, hold_ms=500, cap_ms=1200)
    f = finals(out)
    assert f, "an utterance of endless 'and' must still produce a final"
    assert all(e.held_ms <= 1300 for _, e in f)   # held audio per utterance never exceeds the cap (+1 frame)


def test_disfluency_stripped_before_final_and_uh_alone_is_not_a_turn():
    _, out = run([(1.0, "um how long is the warranty uh", False), (1.5, "um how long is the warranty uh", True)], 3.0)
    assert [e.text for _, e in finals(out)] == ["how long is the warranty"]
    _, out2 = run([(1.0, "uh", False), (1.5, "uh", True)], 3.0)
    assert finals(out2) == []


def test_zh_incomplete_ending_is_held():
    rec, out = run([(1.0, "保修期是", False), (1.5, "保修期是", True)], 3.0, lang="zh")
    f = finals(out)
    assert len(f) == 1 and f[0][0] >= 1.9 and rec.reset_calls == 0


def test_hold_disabled_when_zero():
    _, out = run([(1.0, "how long is the", False), (1.5, "how long is the", True)], 3.0, hold_ms=0)
    assert finals(out)[0][0] <= 1.6


def test_rewritten_tail_does_not_repeat_the_whole_previous_utterance():
    """If the recognizer rewrites the last words after a final was sent, only the genuinely new words must come out."""
    rec, out = run([(1.0, "the cat sat on", False), (1.5, "the cat sat on", True),
                    (2.4, "the kat sat on the mat today", False), (2.9, "the kat sat on the mat today", True)], 4.0)
    texts = [e.text for _, e in finals(out)]
    assert texts[0] == "the cat sat on"
    assert texts[1] == "kat sat on the mat today"[len("kat sat on"):].strip() or texts[1] in ("the mat today", "kat sat on the mat today")
    assert "the kat sat on the mat today" != texts[1]          # never the whole new hypothesis from the start
    assert rec.reset_calls == 0
