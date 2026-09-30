#!/usr/bin/env bash
# Deploy VORA on a Raspberry Pi 4 (64-bit OS) or any Docker host.
#   scripts/deploy_pi.sh            build + run, listen on 127.0.0.1:8000
#   scripts/deploy_pi.sh --https    also create a local CA cert with mkcert and serve HTTPS on the LAN
set -euo pipefail
cd "$(dirname "$0")/.."
command -v docker >/dev/null || { echo "install Docker first: https://docs.docker.com/engine/install/debian/"; exit 1; }
docker compose -f docker/compose.yml build vora
if [[ "${1:-}" == "--https" ]]; then
  command -v mkcert >/dev/null || { echo "install mkcert first (sudo apt install mkcert libnss3-tools)"; exit 1; }
  mkdir -p certs && mkcert -install >/dev/null
  mkcert -cert-file certs/vora.pem -key-file certs/vora-key.pem "$(hostname -I | awk '{print $1}')" "$(hostname).local" localhost
  docker run -d --name vora --restart unless-stopped -p 8443:8000 -v "$PWD/certs:/certs:ro" \
    -e VORA_SSL_CERT=/certs/vora.pem -e VORA_SSL_KEY=/certs/vora-key.pem vora:latest
  echo "open https://$(hostname -I | awk '{print $1}'):8443 (install the mkcert root CA on the client device, mic needs HTTPS)"
else
  docker compose -f docker/compose.yml up -d vora
  echo "open http://127.0.0.1:8000"
fi
