# Deployment

One Dockerfile (`docker/Dockerfile`) for linux/arm64 and linux/amd64, CPU only. Models are fetched at build time from
public Hugging Face repositories at pinned revisions (`scripts/fetch_models.py`, `models/MANIFEST.json`), the index is built
and calibrated in the image. The server binds 0.0.0.0 inside the container; `docker/compose.yml` publishes it on 127.0.0.1 only.

```bash
docker build -f docker/Dockerfile -t vora:arm64 .
```
```bash
docker run --rm -p 127.0.0.1:8000:8000 vora:arm64
```
```bash
.venv/bin/python scripts/ws_smoke.py ws://127.0.0.1:8000/ws client/samples/q_warranty_en.wav
```

`ws_smoke.py` streams a wav at real-time pace over the real WebSocket and exits 0 only if it gets a final transcript,
a context event, audio and the metrics event.

## Targets and what was actually verified
Measured numbers live in `results/deploy.json`; `scripts/report_gates.py` turns them into G7 (PASS only when all four
items below are verified).

| Target | How | Status |
|---|---|---|
| arm64 image (Apple Silicon, Raspberry Pi 4/5 64-bit OS, AWS Graviton, Jetson) | native `docker build` on an M2 | see results/deploy.json `docker_arm64` |
| amd64 image (x86 servers / PCs) | `docker buildx build --platform linux/amd64` under emulation on the M2: build + boot + ws_smoke (latency under emulation is meaningless) | `docker_amd64` |
| Pi-4-shaped container | `docker run --cpus=4 --memory=1g` on the M2 — the cores are much faster than a Pi's, so this only proves the memory limit and the 4-core settings work | `limited_core_run` |
| Pi-class CPU | AWS `a1.xlarge` (Graviton1: 4× Cortex-A72 @ 2.3 GHz, the Pi 4 core at a higher clock), `scripts/pi_bench.sh` inside the image; Pi 4 runs the same core at 1.5 GHz, so expect roughly 1.5× slower — an estimate, not a Pi measurement. Runbook: docs/aws-a1-runbook.md | `pi_class_measured` |
| Real Raspberry Pi 4 | `scripts/deploy_pi.sh`, then `scripts/pi_bench.sh` | **not done (no hardware)** |
| Jetson (Nano / Orin) | same arm64 image, CPU only — no CUDA path exists in VORA. The original Jetson Nano ships JetPack 4 (Ubuntu 18.04, kernel 4.9); the image brings its own userland, so it should run under Docker there, but this is untested | **UNVERIFIED (no hardware)** |

Speculative turns (`VORA_SPECULATE`) switch themselves off below 6 cores, so a 4-core Pi runs without them.

## Phones and tablets on the LAN
See docs/ui.md (TLS certificate via `scripts/make_cert.sh`, `VORA_HOST=0.0.0.0`). Never expose the server to the internet:
there is no authentication.

## Licences inside the image
See docs/licenses.md. Note: the image contains `espeak-ng-data` (GPL-3.0) and the zh voice whose dataset licence is
unknown — fine for the demo, review before redistributing the image.
