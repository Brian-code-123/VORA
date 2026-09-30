import time
import wave
from pathlib import Path

import numpy as np
import pytest
from jiwer import wer

from vora.asr import AsrSession, load_recognizers
from vora.config import Settings

S = Settings()
pytestmark = pytest.mark.skipif(not (S.models_dir / "asr_en").exists(), reason="models not fetched")
SR = 16000
CH = int(0.32 * SR)


@pytest.fixture(scope="module")
def recs():
    return load_recognizers(S)


def read(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        assert w.getframerate() == SR
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def stream_wav(sess, pcm, tail_s=1.5, chunk=CH):
    """Feed pcm then silence; return (events, audio_seconds_fed_at_first_final, per_feed_ms)."""
    x = np.concatenate([pcm, np.zeros(int(tail_s * SR), dtype=np.int16)])
    ev, fed, lat, t_final = [], 0, [], None
    for i in range(0, len(x), chunk):
        c = x[i:i + chunk]
        t0 = time.perf_counter()
        out = sess.feed(c.tobytes())
        lat.append((time.perf_counter() - t0) * 1000)
        fed += len(c)
        ev += out
        if t_final is None and any(e.kind == "final" for e in out):
            t_final = fed / SR
    return ev, t_final, lat


def test_en_wav_streams_partials_then_final(recs):
    p = S.models_dir / "asr_en/test_wavs"
    pcm = read(p / "0.wav")
    ev, _, _ = stream_wav(AsrSession(recs, "en"), pcm)
    parts = [e for e in ev if e.kind == "partial"]
    finals = [e for e in ev if e.kind == "final"]
    assert len(parts) >= 2 and len(finals) == 1
    ref = dict(l.split(" ", 1) for l in (p / "trans.txt").read_text().splitlines())["0.wav"].lower()
    assert wer(ref, finals[0].text) <= 0.15
    assert len(parts[-1].text) > len(parts[0].text)


def test_zh_wav_final_has_cjk(recs):
    pcm = read(S.models_dir / "asr_zh/test_wavs/0.wav")
    ev, _, _ = stream_wav(AsrSession(recs, "zh"), pcm)
    finals = [e for e in ev if e.kind == "final"]
    assert finals and all(any("\u4e00" <= ch <= "\u9fff" for ch in f.text) for f in finals)  # bundled clip pauses mid-sentence -> 2 finals


def test_silence_emits_no_final(recs):
    ev, _, _ = stream_wav(AsrSession(recs, "en"), np.zeros(SR * 3, dtype=np.int16), tail_s=0)
    assert ev == []


def test_text_lowercased_and_stripped(recs):
    pcm = read(S.models_dir / "asr_en/test_wavs/0.wav")
    ev, _, _ = stream_wav(AsrSession(recs, "en"), pcm)
    for e in ev:
        assert e.text == e.text.lower().strip()


def test_endpoint_within_700ms_of_speech_end(recs):
    pcm = read(S.models_dir / "asr_en/test_wavs/0.wav")
    _, t_final, _ = stream_wav(AsrSession(recs, "en"), pcm, tail_s=2.0, chunk=int(0.1 * SR))
    assert t_final is not None
    assert t_final - len(pcm) / SR <= 0.7


def test_30s_monologue_force_final(recs):
    pcm = read(S.models_dir / "asr_en/test_wavs/1.wav")
    reps = int(np.ceil(32 * SR / len(pcm)))
    long = np.tile(pcm, reps)  # continuous speech, no long pause
    ev, _, _ = stream_wav(AsrSession(recs, "en"), long, tail_s=0)
    assert any(e.kind == "final" for e in ev)


@pytest.mark.perf
def test_chunk_decode_under_300ms(recs):
    pcm = read(S.models_dir / "asr_en/test_wavs/1.wav")
    _, _, lat = stream_wav(AsrSession(recs, "en"), pcm)
    assert np.percentile(lat, 95) < 300


def test_second_utterance_in_same_session_is_not_clipped(recs):
    """The 0.8 s leading-silence pre-roll must also follow a reset, not only the session start."""
    p = S.models_dir / "asr_en/test_wavs"
    a, b = read(p / "0.wav"), read(p / "1.wav")
    ref = dict(l.split(" ", 1) for l in (p / "trans.txt").read_text().splitlines())["1.wav"].lower()
    gap = np.zeros(int(0.5 * SR), dtype=np.int16)   # user starts the next question right after the endpoint
    sess = AsrSession(recs, "en")
    ev, _, _ = stream_wav(sess, np.concatenate([a, gap, b]), tail_s=1.5, chunk=int(0.1 * SR))
    finals = [e.text for e in ev if e.kind == "final"]
    assert len(finals) >= 2
    assert wer(ref, " ".join(finals[1:])) <= 0.15, finals
