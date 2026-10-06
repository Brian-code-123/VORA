#!/usr/bin/env bash
# Source tarball for scripts/aws_deploy.sh: exactly the committed tree at HEAD (git archive), so models, data, the venv,
# certificates and local settings can never slip in. Usage: scripts/pack_src.sh [--allow-dirty] [OUT]   (OUT default
# ./vora-src.tar.gz). Refuses uncommitted changes, which would otherwise be silently missing from the box.
set -euo pipefail
ALLOW_DIRTY=0
[ "${1:-}" = "--allow-dirty" ] && { ALLOW_DIRTY=1; shift; }
OUT=$(cd "$(dirname "${1:-vora-src.tar.gz}")" && pwd)/$(basename "${1:-vora-src.tar.gz}")
cd "${PACK_REPO:-$(dirname "$0")/..}"
if [ -n "$(git status --porcelain --untracked-files=no)" ] && [ $ALLOW_DIRTY -eq 0 ]; then
  echo "uncommitted changes: commit them first (the box builds HEAD only), or pass --allow-dirty" >&2
  exit 1
fi
git archive --format=tar.gz -o "$OUT" HEAD
echo "$OUT: $(du -h "$OUT" | cut -f1), $(git rev-parse --short HEAD)"
