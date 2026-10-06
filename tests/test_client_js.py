"""Runs the client's Node tests (state machine, stats, net, env, i18n, contrast, worklet) from the Python suite."""
import shutil
import subprocess
from pathlib import Path

import pytest

CLIENT = Path(__file__).resolve().parent.parent / "client"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_client_node_tests_pass():
    files = sorted(f"tests/{p.name}" for p in (CLIENT / "tests").glob("*.test.mjs"))
    r = subprocess.run(["node", "--test", *files], cwd=CLIENT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]


def test_client_tests_are_passed_as_explicit_files(tmp_path, monkeypatch):
    """Node 22 (the CI runner) does not recurse `node --test tests/` (Cannot find module .../client/tests); Node 18 does.
    Explicit file names behave the same on both, so the wrapper must pass them."""
    args = tmp_path / "args.txt"
    fake = tmp_path / "node"
    fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{args}"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{__import__('os').environ['PATH']}")
    test_client_node_tests_pass()
    got = args.read_text().split()
    files = sorted(f"tests/{p.name}" for p in (CLIENT / "tests").glob("*.test.mjs"))
    assert len(files) >= 4 and got == ["--test", *files], got
