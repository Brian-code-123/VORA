"""Docker was never built in this project; these checks catch the mistakes a build would (a missing COPY source, a
file the .dockerignore removes, a script the image needs but does not copy)."""
import fnmatch
import re
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (ROOT / "docker" / "Dockerfile").read_text()


def copy_sources():
    out = []
    for line in DOCKERFILE.splitlines():
        m = re.match(r"\s*COPY\s+(?:--from=\S+\s+)?(.+)", line)
        if m and "--from" not in line:
            parts = m.group(1).split()
            out += parts[:-1]          # last token is the destination
    return out


def ignored(path: str) -> bool:
    pats = [l.strip().rstrip("/") for l in (ROOT / ".dockerignore").read_text().splitlines() if l.strip() and not l.startswith("#")]
    first = path.strip("./").split("/")[0]
    return any(fnmatch.fnmatch(path.strip("./"), p) or fnmatch.fnmatch(first, p) for p in pats)


def test_dockerfile_copy_sources_exist_and_not_dockerignored():
    srcs = copy_sources()
    assert srcs, "no COPY lines parsed"
    for s in srcs:
        assert (ROOT / s).exists(), f"COPY source missing: {s}"
        assert not ignored(s), f"COPY source removed by .dockerignore: {s}"


def test_dockerfile_installs_onnx_and_quantize_script():
    assert "scripts/quantize_tts.py" in DOCKERFILE and "scripts/fetch_models.py" in DOCKERFILE
    deps = (ROOT / "pyproject.toml").read_text()
    assert '"onnx"' in deps and '"cn2an"' in deps          # quantize + zh number reading need them at runtime
    assert "kb" in " ".join(copy_sources()) and "client" in " ".join(copy_sources())


def test_deploy_script_bash_n():
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts" / "deploy_pi.sh")]).returncode == 0
    assert subprocess.run(["bash", "-n", str(ROOT / "tests" / "test_docker_smoke.sh")]).returncode == 0
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts" / "pi_bench.sh")]).returncode == 0


def test_compose_ports_loopback_only():
    for line in (ROOT / "docker" / "compose.yml").read_text().splitlines():
        if re.search(r"ports:\s*\[", line) or re.match(r"\s*-\s*\"?\d+:\d+", line):
            assert "127.0.0.1:" in line, line


def test_fetch_models_dry_run_lists_pinned_revisions():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_models.py"), "--dry-run"], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    lines = [l for l in out.stdout.splitlines() if "@" in l]
    assert len(lines) >= 6
    for l in lines:
        assert re.search(r"@[0-9a-f]{40}$", l), l


def test_healthcheck_and_loopback_default():
    assert "HEALTHCHECK" in DOCKERFILE and "/health" in DOCKERFILE
    assert "VORA_HOST=0.0.0.0" in DOCKERFILE      # inside the container only; compose publishes on 127.0.0.1
