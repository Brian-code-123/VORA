import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scripts import eval_suite as E
from scripts import refvad

SR = 16000
INTENTS = {"balance": {"chunk_ids": ["B01", "BZ01"], "kw": {"en": ["balance"], "zh": ["余额"]}},
           "freeze": {"chunk_ids": ["B02", "BZ02"], "kw": {"en": ["freeze"], "zh": ["冻结"]}}}


def row(i: int, text: str, intent: str | None = "balance", lang: str = "en", split_hint: str | None = None) -> dict:
    return {"id": f"s_{i}", "wav": f"wav/s_{i}.wav", "text": text, "lang": lang, "scenario": "telephone", "accent": "en-US",
            "intent": intent, "source": "x", "licence": "y", "split": split_hint or ("dev" if i % 2 == 0 else "test"), "orig_sr": 8000, "aug": "clean"}


def pcm(seconds: float = 1.0, amp: int = 4000) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 300 * t)).astype(np.int16)


# ---------- statistics ----------
def test_wilson_bounds_known_values():
    lo, hi = E.wilson(50, 100)
    assert abs(lo - 0.404) < 0.002 and abs(hi - 0.596) < 0.002
    assert E.wilson(0, 10)[0] == 0.0 and abs(E.wilson(0, 10)[1] - 0.2775) < 0.002
    assert E.wilson(10, 10)[1] == 1.0
    assert E.wilson(0, 0) == (0.0, 1.0)                       # no data = no information, never a crash


def test_wer_ci_brackets_the_point_estimate():
    errs, lens = [1, 0, 2, 0, 1, 3, 0, 1], [10, 8, 12, 9, 10, 11, 7, 10]
    lo, hi = E.wer_ci(errs, lens)
    assert lo <= sum(errs) / sum(lens) <= hi and E.wer_ci(errs, lens) == (lo, hi)   # deterministic


# ---------- split discipline ----------
def test_split_test_requires_final_flag():
    rows = [row(i, "t") for i in range(10)]
    assert all(r["split"] == "dev" for r in E.pick(rows, "dev", final=False))
    with pytest.raises(PermissionError, match="--final"):
        E.pick(rows, "test", final=False)
    assert all(r["split"] == "test" for r in E.pick(rows, "test", final=True))
    assert len(E.pick(rows, "dev", final=False, n=2)) == 2


# ---------- augmentation specs ----------
def test_parse_aug_and_unknown():
    assert E.parse_aug("clean") == ("clean", None)
    assert E.parse_aug("ambient@10") == ("ambient", 10.0)
    assert E.parse_aug("reverb@0.6") == ("reverb", 0.6)
    assert E.parse_aug("quiet") == ("quiet", None)
    for bad in ("ambient", "ambient@x", "nope", "babble@"):
        with pytest.raises(ValueError):
            E.parse_aug(bad)


def test_apply_aug_deterministic_and_shape_preserving():
    x = pcm(1.0)
    noise = [pcm(0.5, 1500)]
    others = {"s_1": pcm(1.0, 3000), "s_2": pcm(1.0, 2000), "s_3": pcm(1.0, 2500), "s_4": pcm(1.0, 1800), "s_0": x}
    for spec in ("clean", "ambient@10", "babble@10", "reverb@0.4", "quiet", "loud", "codec"):
        a = E.apply_aug(x, spec, "s_0", list(others), others.__getitem__, noise)
        b = E.apply_aug(x, spec, "s_0", list(others), others.__getitem__, noise)
        assert a.dtype == np.int16 and len(a) == len(x) and np.array_equal(a, b), spec
    assert np.array_equal(E.apply_aug(x, "clean", "s_0", [], None, []), x)
    assert not np.array_equal(E.apply_aug(x, "ambient@10", "s_0", [], None, noise), x)


def test_babble_never_contains_the_target_clip():
    x = pcm(1.0, 4000)
    called = []
    pool = {f"s_{i}": pcm(1.0, 1000 + i) for i in range(1, 8)}
    pool["s_0"] = x

    def loader(i):
        called.append(i)
        return pool[i]
    E.apply_aug(x, "babble@10", "s_0", list(pool), loader, [])
    assert "s_0" not in called and len(called) == 4


# ---------- ASR scoring ----------
def test_score_asr_en_and_zh():
    s = E.score_asr(["what is my balance", "freeze my card"], ["what is my balance", "freeze card"], "en")
    assert s["errors"] == 1 and s["ref_len"] == 7 and abs(s["value"] - 1 / 7) < 1e-9 and s["metric"] == "wer"
    z = E.score_asr(["我想查询余额"], ["我想查询余"], "zh")
    assert z["metric"] == "cer" and z["errors"] == 1 and z["ref_len"] == 6


def test_empty_asr_text_is_a_miss_not_a_crash():
    rows = [row(1, "what is my balance")]
    res = E.evaluate(rows, lambda r: pcm(), lambda rw, x: "", retriever=SimpleNamespace(search=lambda q: []), intents=INTENTS)
    assert res["asr"]["errors"] == 4 and res["retrieval"]["top3"] == 0.0 and res["retrieval"]["refused"] == 1


def test_empty_reference_rows_are_skipped_and_counted():
    rows = [row(1, "what is my balance"), row(2, "   ")]
    res = E.evaluate(rows, lambda r: pcm(), lambda rw, x: "what is my balance")
    assert res["n"] == 1 and res["n_skipped_empty_ref"] == 1


# ---------- retrieval ----------
class FakeRetriever:
    def __init__(self, mapping):
        self.mapping = mapping

    def search(self, q):
        return [SimpleNamespace(chunk_id=c, text=c, score=1.0) for k, cs in self.mapping.items() if k in q.lower() for c in cs]


def test_retrieval_top1_top3_twin_ids_and_ref_vs_asr_text():
    rows = [row(1, "what is my balance"), row(2, "please freeze my card", intent="freeze")]
    hyps = {"what is my balance": "what is my balance", "please freeze my card": "please my card"}   # ASR dropped "freeze"
    r = FakeRetriever({"balance": ["B01"], "freeze": ["B02"]})
    res = E.evaluate(rows, lambda rw: pcm(), lambda rw, x: hyps[rw["text"]], retriever=r, intents=INTENTS)
    assert res["retrieval"]["n"] == 2 and res["retrieval"]["top3"] == 0.5 and res["retrieval"]["top3_ref"] == 1.0
    assert res["retrieval"]["top1"] == 0.5 and res["retrieval"]["refused"] == 1


def test_offtopic_false_accept_rate():
    off = [row(i, f"the quick brown fox number {i}", intent=None) for i in range(4)]
    r = FakeRetriever({"fox number 0": ["B09"]})            # accepts 1 of 4 unrelated sentences
    assert E.offtopic_false_accept(off, r) == 0.25


def test_faith_check_is_case_insensitive_any_keyword():
    assert E.faith_check("Your BALANCE is shown in the app", ["balance"])
    assert E.faith_check("余额在应用首页", ["余额", "balance"])
    assert not E.faith_check("I'm not sure about that.", ["balance"])
    assert not E.faith_check("anything", [])


# ---------- latency ----------
def test_latency_split_answered_vs_refused():
    turns = [{"first_content_audio_ms": 1000.0, "context": ["B01"]}, {"first_content_audio_ms": 1200.0, "context": ["B02"]},
             {"first_content_audio_ms": 1400.0, "context": ["B03"]}, {"first_content_audio_ms": 400.0, "context": []},
             None]
    s = E.latency_summary(turns)
    assert s["n_answered"] == 3 and s["n_refused"] == 1 and s["n_no_response"] == 1
    assert s["answered"]["p50"] == 1200.0 and s["answered"]["p90"] == 1360.0
    assert s["all"]["p50"] == 1100.0 and abs(s["refusal_rate"] - 0.25) < 1e-9


def test_silence_clip_counts_as_no_response_not_a_hang():
    s = E.latency_summary([None, None])
    assert s["n_no_response"] == 2 and s["answered"] is None and s["all"] is None


# ---------- speech_end cross-check ----------
def test_speech_end_disagreement_flagged_above_150ms():
    rep = E.gap_report([10.0, 40.0, 200.0, 400.0])
    assert rep["n_over_150ms"] == 2 and rep["median_ms"] == 120.0 and rep["flagged"] is True
    assert E.gap_report([10.0, 20.0])["flagged"] is False
    assert E.gap_report([])["flagged"] is False


def test_pipeline_speech_end_on_tone_burst():
    x = np.concatenate([np.random.default_rng(0).normal(0, 20, SR // 2), pcm(1.0, 6000), np.random.default_rng(1).normal(0, 20, SR)]).astype(np.int16)
    assert abs(refvad.pipeline_speech_end(x) - 1.5) <= 0.11


SILERO = refvad.SILERO_PATH


@pytest.mark.skipif(not SILERO.exists(), reason="models/aux/silero_vad.onnx not fetched (python scripts/refvad.py --fetch)")
def test_refvad_and_pipeline_vad_are_comparable_on_real_speech():
    """Measured on the bundled clips: the two rules differ by 0.01-0.97 s (quiet speech tails fall under the pipeline's
    250 RMS floor, silero pads its segments). So the gap is reported per run, and the median must stay below half a second."""
    gaps = []
    for w in sorted(Path("models/asr_en/test_wavs").glob("*.wav"))[:2] + sorted(Path("models/asr_zh_ctc/test_wavs").glob("*.wav"))[:2]:
        x, sr = sf.read(w, dtype="int16")
        assert sr == SR
        x = np.concatenate([np.zeros(SR // 2, np.int16), x, np.zeros(SR, np.int16)])
        a, b = refvad.speech_end(x), refvad.pipeline_speech_end(x)
        assert 0 < a < len(x) / SR and 0 < b < len(x) / SR, w
        gaps.append(abs(a - b) * 1000)
    rep = E.gap_report(gaps)
    assert rep["median_ms"] < 500 and rep["max_ms"] < 1500


def test_latency_summary_adjusts_for_silero_gap():
    turns = [{"first_content_audio_ms": 1500.0, "context": ["B01"], "silero_gap_ms": 300.0},    # silero says speech ended 300 ms later
             {"first_content_audio_ms": 1000.0, "context": ["B02"], "silero_gap_ms": -100.0}]   # ... 100 ms earlier
    s = E.latency_summary(turns)
    assert s["answered_vs_silero"]["p50"] == 1150.0 and s["answered"]["p50"] == 1250.0
    assert "answered_vs_silero" not in E.latency_summary([{"first_content_audio_ms": 1.0, "context": ["x"]}])


# ---------- plumbing ----------
def test_unknown_suite_clear_error():
    with pytest.raises(KeyError, match="unknown suite"):
        E.run("nope", split="dev")


def test_bench_audio_dir_plain_wavs(tmp_path: Path):
    for name, secs in (("a", 1.0), ("b", 2.0)):
        sf.write(tmp_path / f"{name}.wav", pcm(secs), SR, subtype="PCM_16")
    (tmp_path / "texts.jsonl").write_text(json.dumps({"file": "a.wav", "text": "hello there", "lang": "en"}) + "\n")
    rows = E.rows_from_dir(tmp_path, default_lang="zh")
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[0]["text"] == "hello there" and rows[0]["lang"] == "en" and rows[1]["lang"] == "zh" and rows[1]["text"] == ""
    assert rows[0]["licence"] and rows[0]["source"]          # own recordings are labelled, not anonymous
    assert E.load_dir_pcm(tmp_path, rows[1]).dtype == np.int16


def test_bench_suite_zh_not_na():
    pins = json.loads((E.ROOT / "eval" / "suites" / "pins.json").read_text())
    ids = pins["latency_ids"]
    assert len(ids.get("zh", {}).get("test", [])) == 42, "run: python scripts/suites.py build all (writes latency_ids)"
    assert len(ids["en"]["test"]) == 42


def test_suite_result_records_quiet_verdict(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VORA_WRITE_RESULTS", "1")
    res = E.finish({"suite": "minds14_zh", "aug": "clean", "split": "dev"}, {"quiet": True, "max_other_cpu_pct": 3.0, "p90_other_cpu_pct": 2.0},
                   out_dir=tmp_path)
    p = tmp_path / "minds14_zh__clean__dev.json"
    assert p.exists() and json.loads(p.read_text())["quiet"] is True and res["quiet"] is True and "host" in res


def test_suite_run_does_not_write_without_env(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("VORA_WRITE_RESULTS", raising=False)
    E.finish({"suite": "minds14_zh", "aug": "clean", "split": "dev"}, {"quiet": False, "max_other_cpu_pct": 90.0, "p90_other_cpu_pct": 80.0}, out_dir=tmp_path)
    assert not list(tmp_path.iterdir())
