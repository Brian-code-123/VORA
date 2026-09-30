#!/usr/bin/env bash
# Build the image, start it, expect /health 200 within 120 s, then report image size and idle RSS.
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -f docker/Dockerfile -t vora:test .
cid=$(docker run -d -p 127.0.0.1:18000:8000 vora:test)
trap 'docker rm -f "$cid" >/dev/null' EXIT
for i in $(seq 1 60); do
  if curl -fs http://127.0.0.1:18000/health >/dev/null; then
    echo "health OK after $((i*2))s"; docker images vora:test --format 'image size: {{.Size}}'
    docker stats --no-stream --format 'idle RSS: {{.MemUsage}}' "$cid"; exit 0
  fi
  sleep 2
done
echo "health never became 200"; docker logs "$cid" | tail -20; exit 1
