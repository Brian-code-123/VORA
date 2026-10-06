"""README.md must not rot: paths exist, numbers come from results/, every folder and script is mapped, no demo secrets."""
import json
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""


def test_readme_exists():
    assert README, "README.md missing"


def test_readme_paths_exist():
    paths = set(re.findall(r"`((?:scripts|docs|src/vora|client|tests|eval|eval_kb|kb|docker|results)/[^`\s*]+)`", README))
    assert paths, "README names no repo paths"
    missing = sorted(p for p in paths if not (ROOT / p.rstrip("/")).exists())
    assert not missing, missing


def test_readme_python_range_matches_pyproject():
    rng = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
    assert rng in README, f"README should state Python {rng}"


def test_readme_gate_table_matches_gates_final_json():
    for g in json.loads((ROOT / "results" / "gates_final.json").read_text()):
        word = {True: "PASS", False: "FAIL", None: "UNVERIFIED"}[g["ok"]]
        assert re.search(rf"\|\s*{re.escape(g['id'])}\b[^\n]*\b{word}\b", README), f"{g['id']} should read {word} in README.md"


def test_readme_reverb_number_matches_results():
    r = json.loads((ROOT / "results" / "suites" / "matrix" / "minds14_en_us__reverb@0.6__test.json").read_text())["retrieval"]
    top3, refused = round(r["top3"] * 100), round(r["refused_rate"] * 100)
    assert re.search(rf"reverb[^\n]*\b{top3}%[^\n]*\b{refused}%", README, re.I), f"reverb line should say top-3 {top3}% and {refused}% refused"


def _tracked_top_dirs():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    return {p.split("/")[0] for p in out if "/" in p}


def test_readme_repo_map_lists_every_top_level_dir():
    missing = sorted(d for d in _tracked_top_dirs() - {".github", ".claude"} if f"`{d}/`" not in README)
    assert not missing, f"add to the repo map: {missing}"


def test_readme_scripts_map_covers_every_script():
    names = sorted(p.name for p in (ROOT / "scripts").iterdir() if p.suffix in (".py", ".sh") and p.name != "__init__.py")
    missing = [n for n in names if n not in README]
    assert not missing, f"add to the scripts map: {missing}"


def test_readme_has_no_ip_or_key():
    ips = set(re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", README)) - {"127.0.0.1", "0.0.0.0"}
    assert not ips, f"public IP in README: {ips}"
    assert not re.search(r"\b[0-9a-f]{24}\b", README), "24-hex string (access key?) in README"
    assert "key=" not in README.replace("?key=…", ""), "query-string key in README"


def test_license_is_apache_2():
    head = (ROOT / "LICENSE").read_text().lstrip()[:200]
    assert head.startswith("Apache License") and "Version 2.0" in head


def test_pyproject_has_no_readme_field():
    """docker/Dockerfile copies only pyproject.toml and src/ before `pip install .`: a readme field would break the build."""
    assert "readme" not in tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
