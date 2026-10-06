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
            parts = [x for x in m.group(1).split() if not x.startswith("--")]      # flags such as --chown=
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


def test_dockerignore_excludes_local_data_keys_and_dev_only_files():
    from pathlib import Path
    ignored = set(Path(".dockerignore").read_text().split())
    assert {"data", "results", "models", "index", "index_bank", "certs", ".claude", "client/tests"} <= ignored   # private keys never in an image


def test_dockerfile_copies_what_ingest_needs():
    from pathlib import Path
    df = Path("docker/Dockerfile").read_text()
    flat = re.sub(r"COPY(?:\s+--\S+)*", "COPY", df)         # ignore flags such as --chown=
    assert "COPY eval ./eval" in flat and "COPY kb ./kb" in flat and "vora.rag.ingest" in df   # calibration reads eval/rag_qa + offtopic_dev


def test_ws_smoke_exits_nonzero_on_no_audio_and_on_no_server(tmp_path):
    import subprocess, sys, wave
    w = tmp_path / "s.wav"
    with wave.open(str(w), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(8000); f.writeframes(b"\x00\x00" * 800)
    from scripts.ws_smoke import check, load_pcm16
    assert len(load_pcm16(str(w))) == 1600 * 2                     # 8 kHz -> 16 kHz
    assert check({"events": [{"type": "final"}, {"type": "context"}, {"type": "metrics"}], "audio_bytes": 0}) == ["audio"]
    r = subprocess.run([sys.executable, "scripts/ws_smoke.py", "ws://127.0.0.1:9/ws", str(w), "--timeout", "2"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2                                        # could not connect


# ---- the container must not run as root (OWASP A05) ----------------------------------------------------------------
def final_stage_lines():
    lines = DOCKERFILE.splitlines()
    starts = [i for i, l in enumerate(lines) if re.match(r"\s*FROM\s", l)]
    return [l.strip() for l in lines[starts[-1]:] if l.strip() and not l.strip().startswith("#")]


def test_final_stage_runs_as_non_root():
    stage = final_stage_lines()
    users = [i for i, l in enumerate(stage) if l.startswith("USER ")]
    assert users, "the final stage never leaves root"
    last = stage[users[-1]].split()[1]
    assert last.split(":")[0] not in ("root", "0"), last
    cmd = next(i for i, l in enumerate(stage) if l.startswith("CMD "))
    assert users[-1] < cmd


def test_copies_after_user_are_chowned_to_it():
    stage = final_stage_lines()
    first_user = next(i for i, l in enumerate(stage) if l.startswith("USER "))
    plain = [l for l in stage[first_user:] if l.startswith("COPY ") and "--chown=" not in l and "--from=" not in l]
    assert not plain, f"these would be root-owned and unwritable for the app user: {plain}"


def test_no_recursive_chown_layer():
    """`chown -R /app` after the models are downloaded would copy 1.2 GB into a new layer."""
    assert not re.search(r"chown\s+-R", DOCKERFILE)


def test_models_are_downloaded_as_the_app_user():
    stage = final_stage_lines()
    user = next(i for i, l in enumerate(stage) if l.startswith("USER "))
    fetch = next(i for i, l in enumerate(stage) if "fetch_models.py" in l and l.startswith("RUN "))
    assert user < fetch, "downloaded models would be root-owned"


def test_aws_deploy_logs_container_uid():
    assert "docker exec vora id -u" in (ROOT / "scripts" / "aws_deploy.sh").read_text()


def test_aws_deploy_prints_certificate_fingerprint():
    """The demo certificate is self-signed, so the browser warning is the only check a visitor gets. The box prints the
    certificate's SHA-256 fingerprint to its console, and docs/deploy.md has the owner compare it before clicking through."""
    sh = (ROOT / "scripts" / "aws_deploy.sh").read_text()
    assert 'say "cert $(openssl x509 -in /opt/vora/certs/vora.crt -noout -fingerprint -sha256)"' in sh
    assert sh.index("openssl req -x509") < sh.index("-fingerprint -sha256") < sh.index("docker run -d --name vora")
    assert "-fingerprint -sha256" in (ROOT / "docs" / "deploy.md").read_text()
