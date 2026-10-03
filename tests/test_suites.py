import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import soundfile as sf

from scripts import suites as S

SR = 16000


def wav_bytes(x: np.ndarray, sr: int) -> bytes:
    b = io.BytesIO()
    sf.write(b, x, sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


def tone8k(seconds: float = 1.5) -> np.ndarray:
    t = np.arange(int(8000 * seconds)) / 8000
    return (3000 * np.sin(2 * np.pi * 400 * t)).astype(np.int16)


def fake_table(n: int = 6, with_bad: bool = False) -> pa.Table:
    rows = []
    for i in range(n):
        rows.append({"path": f"en-US~BALANCE/clip{i}.wav", "audio": {"bytes": wav_bytes(tone8k(), 8000), "path": None},
                     "transcription": f"what is my balance {i}", "english_transcription": f"what is my balance {i}",
                     "intent_class": i % 2, "lang_id": 0})
    if with_bad:
        rows.append({"path": "x/bad.wav", "audio": {"bytes": b"not a wav", "path": None}, "transcription": "bad", "english_transcription": "bad", "intent_class": 0, "lang_id": 0})
        rows.append({"path": "x/empty.wav", "audio": {"bytes": wav_bytes(tone8k(), 8000), "path": None}, "transcription": "  ", "english_transcription": "", "intent_class": 0, "lang_id": 0})
        rows.append({"path": "x/short.wav", "audio": {"bytes": wav_bytes(tone8k(0.3), 8000), "path": None}, "transcription": "hi", "english_transcription": "hi", "intent_class": 0, "lang_id": 0})
        rows.append({"path": "x/long.wav", "audio": {"bytes": wav_bytes(tone8k(21), 8000), "path": None}, "transcription": "long one", "english_transcription": "long one", "intent_class": 0, "lang_id": 0})
    return pa.Table.from_pylist(rows)


def test_split_deterministic_and_balanced():
    ids = [f"minds14_en-US_{i:05d}" for i in range(2000)]
    first = [S.split_of(i) for i in ids]
    assert first == [S.split_of(i) for i in ids]
    assert set(first) == {"dev", "test"}
    assert 0.45 < first.count("dev") / len(first) < 0.55


def test_resample_8k_to_16k_length_and_peak():
    x = tone8k(1.0)
    y = S.to_16k(x, 8000)
    assert y.dtype == np.int16 and abs(len(y) - 16000) <= 1
    assert abs(int(np.abs(y).max()) - int(0.5 * 32767)) <= 2      # peak-normalised to -6 dBFS like the other eval sets


def test_to_16k_silence_stays_silence():
    y = S.to_16k(np.zeros(8000, np.int16), 8000)
    assert not y.any()


def test_decode_audio_garbage_is_none():
    assert S.decode_audio(b"garbage") is None
    x, sr = S.decode_audio(wav_bytes(tone8k(), 8000))
    assert sr == 8000 and len(x) == 12000


def test_manifest_schema_complete():
    row = S.make_row(id="a", wav="wav/a.wav", text="hi", lang="en", scenario="telephone", accent="en-US", intent="balance",
                     source="PolyAI/minds14", licence="CC-BY-4.0", orig_sr=8000)
    assert set(S.FIELDS) <= set(row) and row["aug"] == "clean" and row["split"] in ("dev", "test")


def test_every_row_has_source_and_licence():
    with pytest.raises(ValueError):
        S.make_row(id="a", wav="w", text="t", lang="en", scenario="x", accent="", intent=None, source="", licence="CC-BY-4.0", orig_sr=16000)
    with pytest.raises(ValueError):
        S.make_row(id="a", wav="w", text="t", lang="en", scenario="x", accent="", intent=None, source="s", licence="", orig_sr=16000)


def test_minds14_rows_and_filter_counts_reported(tmp_path: Path):
    rows, report = S.rows_from_minds14(fake_table(6, with_bad=True), "en-US", ["balance", "joint_account"], tmp_path)
    assert len(rows) == 6
    assert report == {"seen": 10, "kept": 6, "corrupt": 1, "empty_text": 1, "too_short": 1, "too_long": 1}
    r = rows[0]
    assert r["lang"] == "en" and r["accent"] == "en-US" and r["scenario"] == "telephone" and r["orig_sr"] == 8000
    assert r["intent"] in ("balance", "joint_account") and r["licence"] == "CC-BY-4.0"
    x, sr = sf.read(tmp_path / r["wav"], dtype="int16")
    assert sr == 16000 and x.ndim == 1


def test_duplicate_file_names_across_intents_get_unique_ids(tmp_path: Path):
    """en-AU names every file response_N.wav inside one folder per intent (en-AU~PAY_BILL/response_4.wav): the first build
    used the file stem as id, 624 rows collapsed to 146 ids and later wavs overwrote earlier ones (94.8% 'WER' was an artefact)."""
    rows_in = []
    for intent_dir, cls in (("en-AU~PAY_BILL", 0), ("en-AU~BALANCE", 1)):
        rows_in.append({"path": f"{intent_dir}/response_4.wav", "audio": {"bytes": wav_bytes(tone8k(), 8000), "path": None},
                        "transcription": f"text for {intent_dir}", "english_transcription": "x", "intent_class": cls, "lang_id": 2})
    rows, _ = S.rows_from_minds14(pa.Table.from_pylist(rows_in), "en-AU", ["pay_bill", "balance"], tmp_path)
    assert len({r["id"] for r in rows}) == 2 and len({r["wav"] for r in rows}) == 2
    assert len(list((tmp_path / "wav").glob("*.wav"))) == 2


def test_duplicate_ids_are_rejected(tmp_path: Path):
    t = pa.Table.from_pylist([{"path": "a/x.wav", "audio": {"bytes": wav_bytes(tone8k(), 8000), "path": None}, "transcription": "one",
                               "english_transcription": "one", "intent_class": 0, "lang_id": 0}] * 2)
    with pytest.raises(ValueError, match="duplicate id"):
        S.rows_from_minds14(t, "en-US", ["balance"], tmp_path)


def test_corrupt_audio_skipped_not_crash(tmp_path: Path):
    rows, report = S.rows_from_minds14(fake_table(0, with_bad=True), "en-US", ["balance"], tmp_path)
    assert report["corrupt"] == 1 and all(r["id"] != "x" for r in rows)


def test_zh_rows_use_chinese_text_and_zh_lang(tmp_path: Path):
    t = pa.Table.from_pylist([{"path": "zh-CN~BALANCE/c.wav", "audio": {"bytes": wav_bytes(tone8k(), 8000), "path": None},
                               "transcription": "我想查询我的账户余额", "english_transcription": "I want to check my balance", "intent_class": 0, "lang_id": 14}])
    rows, _ = S.rows_from_minds14(t, "zh-CN", ["balance"], tmp_path)
    assert rows[0]["lang"] == "zh" and rows[0]["text"] == "我想查询我的账户余额" and rows[0]["accent"] == "zh-CN"


def fake_rows() -> list[dict]:
    rows = []
    for acc in ("en-US", "en-GB", "en-AU"):
        for intent in range(14):
            for k in range(20):
                rows.append(S.make_row(id=f"minds14_{acc}_{intent}_{k}", wav="w", text="t", lang="en", scenario="telephone", accent=acc,
                                       intent=f"i{intent}", source="PolyAI/minds14", licence="CC-BY-4.0", orig_sr=8000))
    for intent in range(14):
        for k in range(20):
            rows.append(S.make_row(id=f"minds14_zh-CN_{intent}_{k}", wav="w", text="t", lang="zh", scenario="telephone", accent="zh-CN",
                                   intent=f"i{intent}", source="PolyAI/minds14", licence="CC-BY-4.0", orig_sr=8000))
    return rows


def test_latency_subset_per_intent_per_accent():
    rows = fake_rows()
    en = S.latency_subset([r for r in rows if r["lang"] == "en"], split="test")
    assert len(en) == 42
    by = {(r["accent"], r["intent"]) for r in rows if r["id"] in set(en)}
    assert len(by) == 42                                   # exactly one clip per (accent, intent)
    assert all(S.split_of(i) == "test" for i in en)
    zh = S.latency_subset([r for r in rows if r["lang"] == "zh"], split="test")
    assert len(zh) == 42                                   # 3 per intent
    assert S.latency_subset([r for r in rows if r["lang"] == "en"], split="test") == en   # deterministic


def test_latency_subset_short_group_takes_what_exists():
    rows = [r for r in fake_rows() if r["lang"] == "zh" and r["intent"] == "i0"][:2]
    assert len(S.latency_subset(rows, split="test", per_zh=3)) <= 2


def test_build_is_idempotent_cached(tmp_path: Path, monkeypatch):
    calls = {"n": 0}

    def fake_fetch(cfg):
        calls["n"] += 1
        return fake_table(4), ["balance", "joint_account"]
    monkeypatch.setattr(S, "_fetch_minds14", fake_fetch)
    monkeypatch.setattr(S, "SUITES_DIR", tmp_path)
    p1 = S.build("minds14_en_us")
    p2 = S.build("minds14_en_us")
    assert p1 == p2 and calls["n"] == 1
    assert len(S.load_manifest("minds14_en_us")) == 4


def test_unknown_suite_clear_error():
    with pytest.raises(KeyError, match="unknown suite"):
        S.build("nope")


def test_pins_json_lists_licence_for_every_source():
    pins = json.loads((S.ROOT / "eval" / "suites" / "pins.json").read_text())
    for name, p in pins["datasets"].items():
        assert p["licence"] and p["repo"], name


@pytest.mark.parametrize("name", list(S.MINDS14) + list(S.READ))
def test_built_manifests_have_unique_ids_and_existing_wavs(name):
    if not (S.SUITES_DIR / name / "manifest.jsonl").exists():
        pytest.skip(f"suite {name} not built")
    rows = S.load_manifest(name)
    assert len({r["id"] for r in rows}) == len(rows)
    assert all((S.SUITES_DIR / name / r["wav"]).exists() for r in rows[:50])
    assert len({r["wav"] for r in rows}) == len(rows)
