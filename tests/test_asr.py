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


def test_second_final_contains_only_new_words(recs):
    """No reset at an endpoint: the stream keeps its context and each final is only the text since the previous one."""
    p = S.models_dir / "asr_en/test_wavs"
    a, b = read(p / "0.wav"), read(p / "1.wav")
    gap = np.zeros(int(0.8 * SR), dtype=np.int16)
    sess = AsrSession(recs, "en")
    ev, _, _ = stream_wav(sess, np.concatenate([a, gap, b]), tail_s=1.5, chunk=int(0.1 * SR))
    finals = [e.text for e in ev if e.kind == "final"]
    assert len(finals) >= 2                                   # clip 1 has pauses: it may split into several finals
    assert "yellow" in finals[0] and not any("yellow" in f for f in finals[1:])
    assert "consequence" in " ".join(finals[1:]) and "consequence" not in finals[0]
    assert finals[1].startswith("god")                         # leading word no longer clipped after an endpoint


def test_partials_after_a_final_start_fresh(recs):
    p = S.models_dir / "asr_en/test_wavs"
    a, b = read(p / "0.wav"), read(p / "1.wav")
    sess = AsrSession(recs, "en")
    ev, _, _ = stream_wav(sess, np.concatenate([a, np.zeros(int(0.8 * SR), dtype=np.int16), b]), tail_s=1.5, chunk=int(0.1 * SR))
    first_final = next(i for i, e in enumerate(ev) if e.kind == "final")
    later = [e for e in ev[first_final + 1:] if e.kind == "partial"]
    assert later and "yellow" not in later[0].text   # partial display restarts after the final


def test_stream_resets_after_20s_of_audio_at_a_final(recs):
    p = S.models_dir / "asr_en/test_wavs"
    a = read(p / "0.wav")
    sess = AsrSession(recs, "en")
    x = np.concatenate([np.concatenate([a, np.zeros(SR, dtype=np.int16)]) for _ in range(4)])   # ~30 s, 4 utterances
    ev, _, _ = stream_wav(sess, x, tail_s=1.5, chunk=int(0.1 * SR))
    finals = [e.text for e in ev if e.kind == "final"]
    assert len(finals) == 4, finals                           # no spurious one-letter final from trailing noise
    assert sess._stream_samples < 20 * SR


def _session_over_clips(recs, lang, clips, gap_s=1.2):
    """One AsrSession, many utterances in a row (what a real conversation is). Returns (finals-per-clip, session)."""
    sess = AsrSession(recs, lang)
    per_clip, t = [], 0.0
    for pcm in clips:
        x = np.concatenate([pcm, np.zeros(int(gap_s * SR), dtype=np.int16)])
        got = []
        for i in range(0, len(x), 1600):
            got += [e.text for e in sess.feed(x[i:i + 1600].tobytes()) if e.kind == "final"]
        per_clip.append(" ".join(got))
    return per_clip, sess


@pytest.mark.skipif(not (Path(__file__).resolve().parent.parent / "data/eval_cache/en_librispeech_clean.npz").exists(), reason="eval cache missing")
def test_multi_utterance_english_session_keeps_each_final_clean(recs):
    """Beam search rewrites the last words after an endpoint; slicing by `raw.startswith(emitted)` then returned the WHOLE
    text (previous utterance repeated). Eight clips in one session must each yield their own words."""
    import scripts.eval_asr as E
    data = E.load("en_librispeech_clean", 8)
    finals, sess = _session_over_clips(recs, "en", [p for _, p in data])
    refs = [E.norm(r, "en") for r, _ in data]
    hyps = [E.norm(f, "en") for f in finals]
    err = wer(refs, hyps)
    print("multi-utterance en WER:", round(err, 3), "| rewrites:", sess.rewrites)
    assert err <= 0.15, (err, list(zip(refs, hyps)))
