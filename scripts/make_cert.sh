#!/usr/bin/env bash
# Self-signed TLS certificate so phones/tablets on the LAN can use the microphone (browsers require HTTPS off localhost).
# Usage: scripts/make_cert.sh [LAN_IP]   -> certs/vora.crt certs/vora.key
# Then:  VORA_HOST=0.0.0.0 VORA_SSL_CERT=certs/vora.crt VORA_SSL_KEY=certs/vora.key python -m vora.server
# Phones show a certificate warning once (self-signed). Only do this on a network you trust.
set -euo pipefail
cd "$(dirname "$0")/.."
ip="${1:-$( (ipconfig getifaddr en0 || hostname -I | awk '{print $1}') 2>/dev/null)}"
[ -n "$ip" ] || { echo "could not detect the LAN IP; pass it as the first argument" >&2; exit 1; }
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes -days 30 -keyout certs/vora.key -out certs/vora.crt \
  -subj "/CN=vora-local" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:${ip}" 2>/dev/null
chmod 600 certs/vora.key
echo "certs/vora.crt for https://${ip}:8000 (valid 30 days)"
