"""The brief asks for a 2-3 page technical report. Checked on the real PDF, and the print CSS may not cheat with tiny type."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def test_report_word_count_le_1500():
    words = len(re.findall(r"\w+", (ROOT / "docs" / "report.md").read_text(encoding="utf-8")))
    assert words <= 1500, words


def test_report_css_min_font_and_margin():
    css = (ROOT / "docs" / "report.css").read_text()
    body = float(re.search(r"body \{[^}]*font:\s*([\d.]+)pt", css).group(1))
    table = float(re.search(r"table \{[^}]*font-size:\s*([\d.]+)pt", css).group(1))
    margins = [float(x) for x in re.search(r"@page \{[^}]*margin:\s*([\d.]+)mm\s+([\d.]+)mm", css).groups()]
    assert body >= 10 and table >= 9 and min(margins) >= 14


@pytest.mark.integration
@pytest.mark.skipif(not (CHROME.exists() or shutil.which("google-chrome")) or not shutil.which("pdfinfo") or not shutil.which("pandoc"),
                    reason="needs pandoc, Chrome and pdfinfo")
def test_report_pdf_pages_le_3():
    subprocess.run(["bash", "scripts/build_report.sh"], cwd=ROOT, check=True, capture_output=True, timeout=180)
    out = subprocess.run(["pdfinfo", str(ROOT / "docs" / "report.pdf")], capture_output=True, text=True).stdout
    assert int(re.search(r"Pages:\s+(\d+)", out).group(1)) <= 3


def test_report_numbers_match_gates_json():
    import json
    gj = ROOT / "results" / "gates_final.json"
    if not gj.exists():
        pytest.skip("results/gates_final.json not written yet (T8)")
    report = (ROOT / "docs" / "report.md").read_text(encoding="utf-8")
    for g in json.loads(gj.read_text()):
        word = {True: "PASS", False: "FAIL", None: "UNVERIFIED"}[g["ok"]]
        assert re.search(rf"\|\s*{re.escape(g['id'])}\b[^\n]*\b{word}\b", report), f"{g['id']} should read {word} in docs/report.md"
