# Security

VORA is a demo of a streaming voice pipeline, not a hardened product. What protects the hosted demo, and what was knowingly left open, is in the **Security** section of [README.md](README.md); the measured prompt-injection results and the dependency-audit decisions are in [docs/rulings.md](docs/rulings.md).

## Reporting a problem

Please open a GitHub issue for anything that is not exploitable on the hosted demo. If you found a way to use the hosted demo without its access key, or to read its logs, do not publish details: open a private security advisory on this repository (Security tab, "Report a vulnerability") or contact the maintainer through the GitHub profile.

## What to know before running it yourself

- The server listens on `127.0.0.1` by default. Set `VORA_ACCESS_KEY` before making it reachable from other networks: without it anyone who can reach the port can use the models.
- The access key travels as the first WebSocket message and in the link's `#key=` fragment, never in a URL path or query string.
- The demo certificate is self-signed; compare its SHA-256 fingerprint (printed at deploy time) before trusting it.
