import time
import wave
from pathlib import Path

import numpy as np
import pytest

from vora.asr import AsrSession, load_recognizers
from vora.config import Settings

SR = 16000
CTC = Settings().models_dir / "asr_zh_ctc"
needs_ctc = pytest.mark.skipif(not (CTC / "model.int8.onnx").exists(), reason="ctc model not fetched")


def read(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def test_norm_zh_numbers():
    from scripts.eval_asr import norm_zh
    assert norm_zh("桥下垂直净空十五米") == norm_zh("桥下垂直净空15米")
    assert norm_zh("二零一七年八月") == norm_zh("2017年8月")
    assert norm_zh("你好，世界！ ") == "你好世界"


@needs_ctc
def test_ctc_small_size_le_50mb():
    assert (CTC / "model.int8.onnx").stat().st_size <= 50 * 2**20


@needs_ctc
def test_ctc_streams_partials_then_final():
    s = Settings(asr_zh_model="ctc_small")
    recs = load_recognizers(s)
    sess = AsrSession(recs, "zh")
    x = np.concatenate([read(CTC / "test_wavs/0.wav"), np.zeros(int(1.5 * SR), dtype=np.int16)])
    ev = []
    for i in range(0, len(x), 1600):
        ev += sess.feed(x[i:i + 1600].tobytes())
    parts = [e for e in ev if e.kind == "partial"]
    finals = [e for e in ev if e.kind == "final"]
    assert len(parts) >= 2 and finals
    assert all(any("一" <= c <= "鿿" for c in f.text) for f in finals)


@needs_ctc
@pytest.mark.perf
def test_chunk_decode_p95_le_300ms_ctc():
    recs = load_recognizers(Settings(asr_zh_model="ctc_small"))
    sess = AsrSession(recs, "zh")
    x = np.concatenate([read(CTC / "test_wavs/0.wav"), np.zeros(SR, dtype=np.int16)])
    lat = []
    for i in range(0, len(x), 1600):
        t0 = time.perf_counter()
        sess.feed(x[i:i + 1600].tobytes())
        lat.append((time.perf_counter() - t0) * 1000)
    assert np.percentile(lat, 95) < 300, np.percentile(lat, 95)


@needs_ctc
@pytest.mark.perf
def test_endpoint_within_700ms_zh():
    recs = load_recognizers(Settings(asr_zh_model="ctc_small"))
    sess = AsrSession(recs, "zh")
    pcm = read(CTC / "test_wavs/0.wav")
    x = np.concatenate([pcm, np.zeros(2 * SR, dtype=np.int16)])
    fed, t_final = 0, None
    for i in range(0, len(x), 1600):
        out = sess.feed(x[i:i + 1600].tobytes())
        fed += min(1600, len(x) - i)
        if t_final is None and any(e.kind == "final" for e in out):
            t_final = fed / SR
    assert t_final is not None and t_final - len(pcm) / SR <= 0.7


@needs_ctc
@pytest.mark.eval
@pytest.mark.parametrize("model", ["ctc_small"])   # zipformer14m measured 16.7%: rejected, see docs/spikes.md
def test_zh_cer_aishell_le_15pct(model):
    from jiwer import cer
    import scripts.eval_asr as E
    data = E.load("zh_aishell", 60)
    recs = load_recognizers(Settings(asr_zh_model=model))
    refs, hyps = [], []
    for ref, pcm in data:
        h, _ = E.transcribe(recs, "zh", pcm)
        refs.append(E.norm_zh(ref)); hyps.append(E.norm_zh(h))
    err = cer(refs, hyps)
    print(f"AISHELL-1 test CER {model}: {err:.3f} on {len(refs)} clips")
    assert err <= 0.15, err
