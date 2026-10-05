# syntax=docker/dockerfile:1
# Multi-arch (linux/amd64, linux/arm64). CPU only. Models are fetched at build time (public HF repos).
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY pyproject.toml ./
COPY src ./src
# llama-cpp-python has no arm64 wheel: it compiles here (10-20 min under qemu, ~3 min native). Portable CPU build.
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF"
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY --from=build /install /usr/local
WORKDIR /app
ENV VORA_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src
COPY scripts/fetch_models.py scripts/quantize_tts.py scripts/
RUN python scripts/fetch_models.py
COPY src ./src
COPY kb ./kb
COPY eval ./eval
COPY client ./client
RUN python -m vora.rag.ingest
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 CMD curl -fs http://127.0.0.1:8000/health || exit 1
CMD ["python", "-m", "vora.server"]
