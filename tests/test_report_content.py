"""The brief lists what the 2-3 page report must contain. Each test reads docs/report.md and compares it with the
measured files in results/, so a number in the report cannot drift from its source."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
REPORT = (ROOT / "docs" / "report.md").read_text(encoding="utf-8")


def _results(name: str) -> dict:
    path = ROOT / "results" / name
    if not path.exists():
        pytest.skip(f"results/{name} not present")
    return json.loads(path.read_text())


def _section(title_word: str) -> str:
    parts = re.split(r"^## ", REPORT, flags=re.M)
    hit = [p for p in parts if p.splitlines()[0].lower().find(title_word.lower()) >= 0]
    assert hit, f"report has no section about '{title_word}'"
    return hit[0]


def test_report_gives_tts_mos_for_both_voices():
    mos = _results("tts_mos.json")
    assert "MOS" in REPORT and "UTMOS" in REPORT
    for lang in ("en", "zh"):
        assert f"{mos[lang]['mean_mos_proxy']:.2f}" in REPORT, f"{lang} MOS proxy missing from the report"


def test_report_gives_asr_rtf_range_per_language():
    asr = _results("asr.json")
    assert "RTF" in REPORT
    for prefix in ("en_", "zh_"):
        vals = [v["rtf"] for k, v in asr.items() if k.startswith(prefix)]
        span = f"{min(vals):.3f} to {max(vals):.3f}"
        assert span in REPORT, f"{prefix} RTF range '{span}' missing from the report"


def test_report_has_latency_breakdown_that_matches_the_final_bench():
    bench = _results("bench.json")
    rows = [r for r in bench["rows"] if r.get("lang") == "en" and r.get("first_content_audio_ms") is not None]
    stages = {"ASR": "asr_final", "RAG + LLM first token": "rag_first_token",
              "TTS first chunk": "tts_first_chunk", "Total": "first_content_audio_ms"}
    for label, key in stages.items():
        p50, p90 = (round(float(x)) for x in np.percentile([r[key] for r in rows], [50, 90]))
        pattern = rf"^\|\s*{re.escape(label)}[^|\n]*\|\s*{p50}\s*\|\s*{p90}\s*\|"
        assert re.search(pattern, REPORT, flags=re.M), f"breakdown row '{label}' should read {p50} / {p90}"


def test_report_explains_stack_choices_with_a_reason_per_component():
    section = _section("stack")
    for part in ("ASR", "Retrieval", "LLM", "TTS"):
        row = re.search(rf"^\|\s*{part}\b[^\n]*$", section, flags=re.M)
        assert row, f"no stack row for {part}"
        why = row.group(0).strip("|").split("|")[-1].strip()
        assert len(why) >= 40, f"the reason for the {part} choice is too thin: {why!r}"


def test_report_lists_trade_offs():
    section = _section("trade-off")
    assert len(re.findall(r"^- ", section, flags=re.M)) >= 4
