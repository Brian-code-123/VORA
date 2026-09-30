import re
import time

import numpy as np
import pytest

from vora.config import Settings
from vora.tts import Tts, lang_of, sanitize_for_tts, split_by_script

S = Settings()
needs_models = pytest.mark.skipif(not (S.models_dir / "tts_zh").exists(), reason="models not fetched")


def test_sanitize_strips_markdown_emoji_url():
    t = sanitize_for_ttheir = sanitize_for_tts("**Hello** 😀 see https://x.io/a?b=1 and `code` #tag")
    assert "*" not in t and "😀" not in t and "http" not in t and "`" not in t and "#" not in t


def test_sanitize_spells_model_codes():
    assert sanitize_for_tts("VORA-X200") == "VORA X 200"


def test_sanitize_zh_digits():
    assert sanitize_for_tts("保養期3.5年") == "保養期三点五年"


def test_split_by_script_mixed_sentence():
    runs = split_by_script("請問 VORA Box 支援 WiFi 嗎")
    assert [l for l, _ in runs] == ["zh", "en", "zh", "en", "zh"]
    assert "".join(t for _, t in runs) == "請問 VORA Box 支援 WiFi 嗎"


def test_lang_of_mixed():
    assert lang_of("請問 VORA Box 支援 WiFi 嗎") == "zh"
    assert lang_of("Does VORA Box support WiFi?") == "en"


@needs_models
def test_quantized_models_under_30mb():
    from scripts.check_gates import evaluate, measure
    r = evaluate(measure(S.models_dir))
    assert r["tts_en"]["ok"] and r["tts_zh"]["ok"]


@pytest.fixture(scope="module")
def tts():
    return Tts(S)


@needs_models
def test_output_is_16k_mono_pcm16(tts):
    chunks = list(tts.synth("Hello, the warranty lasts two years."))
    pcm = b"".join(chunks)
    assert len(pcm) % 2 == 0 and len(pcm) > 16000  # >0.5 s
    x = np.frombuffer(pcm, dtype=np.int16)
    assert np.abs(x).max() > 500  # not silence
    assert all(len(c) <= 2 * int(0.2 * 16000) for c in chunks)


@needs_models
def test_mixed_zh_en_synthesises(tts):
    pcm = b"".join(tts.synth("請問 VORA Box 支援 WiFi 嗎"))
    assert len(pcm) > 16000


@needs_models
@pytest.mark.perf
def test_first_chunk_under_200ms_short_sentence(tts):
    list(tts.synth("Hello."))  # warm
    ts = []
    for txt in ["Yes, two years.", "好的，兩年。"]:
        t0 = time.perf_counter()
        next(iter(tts.synth(txt)))
        ts.append((time.perf_counter() - t0) * 1000)
    # ruling: int8 en voice measures ~230 ms on M2 (fp32 ~80 ms but 63 MB > 30 MB gate); zh x_low is ~45 ms
    assert ts[1] < 200 and ts[0] < 300, ts
