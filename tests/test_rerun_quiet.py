"""scripts/rerun_quiet.sh: waits for a quiet host and gives up loudly instead of measuring noise."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "rerun_quiet.sh"


def run(**env):
    import os
    return subprocess.run(["bash", str(SCRIPT), "minds14_en_us"], cwd=ROOT, capture_output=True, text=True, timeout=20,
                          env={**os.environ, "QUIET_POLL_S": "0", **env})


def test_exits_2_when_host_never_quiet():
    r = run(QUIET_CHECK="false", QUIET_WAIT_MIN="0")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "never quiet" in r.stdout and "top CPU processes" in r.stdout


def test_gives_up_after_three_busy_tries(tmp_path):
    """The suite run is stubbed by SUITE_CMD; a result that never comes back quiet must end in exit 1, gate UNVERIFIED."""
    r = run(QUIET_CHECK="true", SUITE_CMD="true", RESULTS_DIR=str(tmp_path))
    assert r.returncode == 1, r.stdout + r.stderr
    assert r.stdout.count("try ") == 3 and "no quiet result: minds14_en_us" in r.stdout


def test_script_syntax():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
