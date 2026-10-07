# syntax=docker/dockerfile:1
# Multi-arch (linux/amd64, linux/arm64). CPU only. Models are fetched at build time (public HF repos).
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY pyproject.toml ./
COPY docker/constraints.txt ./constraints.txt
COPY src ./src
# llama-cpp-python has no arm64 wheel: it compiles here (10-20 min under qemu, ~3 min native). Portable CPU build.
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF"
# Install exactly what uv.lock locks (docker/constraints.txt, made by scripts/make_constraints.sh): an unpinned `pip install .` picked up
# llama-cpp-python 0.3.36 the day it was released, it failed to compile on the box, and CI had audited a different set.
RUN pip install --no-cache-dir -c constraints.txt --prefix=/install .

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY --from=build /install /usr/local
# Not root: a bug in the server, a model parser or a dependency then cannot touch the system. Only the empty /app is chowned;
# everything after USER is created by (or --chown'ed to) the app user, so no layer ever duplicates the 1.2 GB of models.
RUN useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin vora && mkdir /app && chown vora:vora /app
WORKDIR /app
ENV VORA_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HOME=/home/vora
USER vora
COPY --chown=vora:vora scripts/fetch_models.py scripts/quantize_tts.py scripts/
RUN python scripts/fetch_models.py
COPY --chown=vora:vora src ./src
COPY --chown=vora:vora kb ./kb
COPY --chown=vora:vora eval ./eval
COPY --chown=vora:vora client ./client
RUN python -m vora.rag.ingest
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 CMD curl -fs http://127.0.0.1:8000/health || exit 1
CMD ["python", "-m", "vora.server"]
