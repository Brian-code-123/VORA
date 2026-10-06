"""scripts/pack_src.sh builds the source tarball that aws_deploy.sh uploads and docker/Dockerfile builds from on the box.
A missing COPY source only shows up 8 minutes into the box's boot, so it is checked here."""
import re
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "pack_src.sh"


@pytest.fixture(scope="module")
def members(tmp_path_factory):
    out = tmp_path_factory.mktemp("pack") / "vora-src.tar.gz"
    r = subprocess.run(["bash", str(SCRIPT), "--allow-dirty", str(out)], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    with tarfile.open(out) as t:
        names = t.getnames()
    return names, out


def test_tarball_has_every_dockerfile_copy_source(members):
    names, _ = members
    srcs = []
    for line in (ROOT / "docker" / "Dockerfile").read_text().splitlines():
        if line.startswith("COPY ") and "--from=" not in line:
            srcs += [x for x in line.split()[1:] if not x.startswith("--")][:-1]      # skip flags such as --chown=
    assert srcs
    missing = [s for s in srcs if not any(n == s or n.startswith(s.rstrip("/") + "/") for n in names)]
    assert not missing, missing


def test_tarball_excludes_models_data_venv_secrets(members):
    names, _ = members
    bad = [n for n in names if re.match(r"(models/(?!(MANIFEST\.json|\.gitignore)$)|data/|\.venv/|index/|index_bank/|certs/)", n)
           or n.endswith((".pem", ".key", "settings.local.json", ".env"))]
    assert not bad, bad[:5]


def test_tarball_member_names_have_no_top_dir_prefix(members):
    names, _ = members
    assert "scripts/aws_deploy.sh" in names and "docker/Dockerfile" in names


def test_tarball_under_5mb(members):
    _, out = members
    assert out.stat().st_size < 5 * 2**20, out.stat().st_size


def test_refuses_dirty_tree(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    g = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
    g("init", "-q")
    (repo / "a.txt").write_text("1")
    g("add", "a.txt")
    g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    (repo / "a.txt").write_text("2")
    r = subprocess.run(["bash", str(SCRIPT), str(tmp_path / "o.tar.gz")], cwd=repo, capture_output=True, text=True,
                       env={**__import__("os").environ, "PACK_REPO": str(repo)})
    assert r.returncode != 0 and "uncommitted" in (r.stdout + r.stderr)
    assert not (tmp_path / "o.tar.gz").exists()
