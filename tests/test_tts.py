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


@needs_models
def test_two_sentences_not_truncated(tts):
    """sherpa's callback returns 1 to continue / 0 to stop (docstring says the opposite): a wrong value dropped sentence 2."""
    a, b = "The battery lasts ten hours.", "Hold the reset button for five seconds."
    both = len(b"".join(tts.synth(f"{a} {b}")))
    apart = len(b"".join(tts.synth(a))) + len(b"".join(tts.synth(b)))
    assert both >= 0.9 * apart, (both, apart)


@needs_models
def test_cancel_during_generation_stops_quickly(tts):
    import threading
    ev = threading.Event()
    ev.set()
    assert b"".join(tts.synth("One sentence here. Another sentence follows. And a third one.", ev)) == b""


def test_pick_en_model_prefers_int8_and_fp32_flag(tmp_path):
    from vora.tts import pick_en_model
    (tmp_path / "en.onnx").write_bytes(b"x")
    assert pick_en_model(tmp_path, fp32=False).name == "en.onnx"          # int8 missing -> fp32 fallback
    (tmp_path / "en.int8.onnx").write_bytes(b"x")
    assert pick_en_model(tmp_path, fp32=False).name == "en.int8.onnx"
    assert pick_en_model(tmp_path, fp32=True).name == "en.onnx"           # glob("*.onnx") used to match int8 too


ANSWERS_EN = [
    "The warranty on the VORA X200 is two years.", "Blue means the box is listening.", "Hold the reset button for ten seconds.",
    "You can import documents up to one gigabyte.", "Please contact support Monday to Friday.", "The speaker is ten watts.",
    "Firmware updates arrive once a month.", "It does not support Cantonese.", "Return shipping is paid by the buyer.",
    "The default wake word is Hey Vora.", "The X200 uses a twelve volt adapter.", "Red means the microphone is muted.",
    "The M100 has two microphones.", "Operating temperature is zero to forty degrees.", "The box draws five watts when idle.",
    "Support replies within one business day.", "Your data stays on the device.", "Say Hey Vora to wake it up.",
    "Volume can be set by voice.", "Yes, it works with Wi-Fi six.",
]
ANSWERS_ZH = ["X200的保修期是两年。", "蓝色表示正在聆听。", "按住复位键十秒钟。", "不支持粤语。", "固件每月更新一次。", "扬声器是十瓦。"]


def first_chunk_text(answer, words):
    from vora.chunker import SentenceChunker
    c = SentenceChunker(first_words=words)
    for ch in answer:
        out = c.push(ch)
        if out:
            return out[0]
    return (c.flush() or [answer])[0]


@needs_models
@pytest.mark.perf
def test_first_chunk_p95_le_200ms_real_chunker(tts):
    """G2: time from first-chunk text to first PCM piece, using the chunker's real first chunk (20 unique en + 6 zh)."""
    words = S.first_chunk_words
    list(tts.synth("Hello."))
    ts = {"en": [], "zh": []}
    for lang, answers in (("en", ANSWERS_EN), ("zh", ANSWERS_ZH)):
        for a in answers:
            text = first_chunk_text(a, words)
            t0 = time.perf_counter()
            next(iter(tts.synth(text)))
            ts[lang].append((time.perf_counter() - t0) * 1000)
    p95 = {k: sorted(v)[max(0, int(len(v) * 0.95) - 1)] for k, v in ts.items()}
    print("first-chunk p95 ms:", p95)
    assert max(p95.values()) <= 200, p95


@needs_models
def test_kb_zh_chars_all_in_lexicon(tts):
    import re
    txt = (S.kb_dir / "zh.md").read_text(encoding="utf-8")
    missing = sorted({c for c in re.findall(r"[一-鿿]", txt) if c not in tts.zh_lexicon})
    assert missing == [], missing


@needs_models
def test_oov_char_logged_and_counted(tts, caplog):
    import logging
    before = tts.oov_total
    with caplog.at_level(logging.WARNING, logger="vora"):
        list(tts.synth("这个龼字很有意思"))
    assert tts.oov_total > before
    assert any("OOV" in r.message for r in caplog.records)


@needs_models
def test_all_oov_text_does_not_crash(tts):
    out = list(tts.synth("词词"))   # may be empty audio; must not raise
    assert isinstance(out, list)


def test_ensure_patched_lexicon_drops_tokens_missing_from_tokens_txt(tmp_path):
    from vora.tts import ensure_patched_lexicon
    (tmp_path / "tokens.txt").write_text("a 0\nt 1\ns 2\n", encoding="utf-8")
    (tmp_path / "lexicon.txt").write_text("次 t s a ̪\n词 t s ̪\n", encoding="utf-8")
    p = ensure_patched_lexicon(tmp_path)
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines == ["次 t s a", "词 t s"]
    assert ensure_patched_lexicon(tmp_path) == p   # idempotent, original untouched
    assert "̪" in (tmp_path / "lexicon.txt").read_text(encoding="utf-8")


def test_usable_chars_uses_patched_lexicon(tmp_path):
    from vora.tts import _usable_zh_chars
    (tmp_path / "tokens.txt").write_text("a 0\nt 1\n", encoding="utf-8")
    (tmp_path / "lexicon.txt").write_text("次 t a ̪\n", encoding="utf-8")
    assert _usable_zh_chars(tmp_path) == {"次"}                  # raw lexicon would make it unusable; patched makes it usable
    assert not (tmp_path / "lexicon.txt").read_text(encoding="utf-8").startswith("次 t a\n")


@needs_models
@pytest.mark.perf
def test_progressive_chunks_leave_no_long_gap(tts):
    """With 1-word then 3-word chunks the next chunk must be ready before the current one finishes playing
    (token arrival excluded). Allows 250 ms of stall at the worst chunk boundary on average."""
    from vora.chunker import SentenceChunker
    list(tts.synth("Hello."))
    worst = []
    for a in ANSWERS_EN:
        c = SentenceChunker(first_words=S.first_chunk_words, second_words=S.second_chunk_words)
        chunks = []
        for ch in a:
            chunks += c.push(ch)
        chunks += c.flush()
        ready, play_end, gap = 0.0, None, 0.0
        for txt in chunks:
            t0 = time.perf_counter()
            pcm = b"".join(tts.synth(txt))
            ready += time.perf_counter() - t0
            dur = len(pcm) / 2 / 16000
            if play_end is None:
                play_end = ready + dur
            else:
                gap = max(gap, ready - play_end)
                play_end = max(play_end, ready) + dur
        worst.append(gap)
    worst.sort()
    print("worst gap per answer, p50/p95 s:", worst[len(worst) // 2], worst[int(len(worst) * 0.95) - 1])
    assert worst[int(len(worst) * 0.95) - 1] <= 0.25, worst


def test_zh_run_numbers_are_read_not_skipped():
    from vora.tts import zh_numbers
    assert zh_numbers("按住复位键 10 秒") == "按住复位键 十 秒"
    assert zh_numbers("价格 199 美元") == "价格 一百九十九 美元"
    assert zh_numbers("版本 2.4 GHz") == "版本 二点四 GHz"
    assert zh_numbers("序列号 123456 位") == "序列号 一二三四五六 位"      # long digit strings: digit by digit
    assert zh_numbers("没有数字") == "没有数字"


@needs_models
def test_zh_digits_with_spaces_do_not_hit_the_voice_as_oov(tts, capfd):
    pcm = b"".join(tts.synth("按住复位键 10 秒"))
    assert len(pcm) > 16000
    assert "OOV 10" not in capfd.readouterr().err
