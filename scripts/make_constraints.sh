#!/usr/bin/env bash
# Regenerate docker/constraints.txt from uv.lock: the image installs exactly these versions (the runtime dependencies only,
# no dev or eval extras). Run it after every change to uv.lock; tests/test_dockerfile_static.py fails when the two disagree.
set -euo pipefail
cd "$(dirname "$0")/.."
uv export --locked --no-hashes --no-emit-project -o docker/constraints.txt
echo "docker/constraints.txt: $(grep -c '==' docker/constraints.txt) pinned packages"
