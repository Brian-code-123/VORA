"""Runs the client's Node tests (state machine, stats, net, env, i18n, contrast, worklet) from the Python suite."""
import shutil
import subprocess
from pathlib import Path

import pytest

CLIENT = Path(__file__).resolve().parent.parent / "client"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_client_node_tests_pass():
    r = subprocess.run(["node", "--test", "tests/"], cwd=CLIENT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
