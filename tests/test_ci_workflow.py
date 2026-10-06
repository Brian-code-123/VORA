"""CI must audit dependencies (OWASP A06), and every vulnerability it is told to ignore needs a written reason."""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def audit_step():
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    steps = [s for job in wf["jobs"].values() for s in job["steps"]]
    found = [s for s in steps if "pip-audit" in str(s.get("run", ""))]
    assert found, "ci.yml has no pip-audit step"
    return found[0]


def test_ci_audits_dependencies():
    step = audit_step()
    assert "uv export --locked" in step["run"], "audit the committed lock, and fail when it no longer matches pyproject.toml"
    assert not step.get("continue-on-error"), "an audit that cannot fail audits nothing"


def test_every_ignored_vulnerability_is_explained_in_rulings():
    ids = re.findall(r"--ignore-vuln\s+(\S+)", audit_step()["run"])
    rulings = (ROOT / "docs" / "rulings.md").read_text(encoding="utf-8")
    missing = [i for i in ids if i not in rulings]
    assert not missing, f"add a ruling (why it does not apply, when to review) for: {missing}"
