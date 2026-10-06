"""Security headers on every HTTP response (OWASP A05). The CSP only works because the page has no inline script or style."""
import re
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from tests.test_server import models, wait_ready
from vora.config import Settings
from vora.server import create_app

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = {"x-content-type-options": "nosniff", "referrer-policy": "no-referrer", "permissions-policy": "microphone=(self)"}
CSP_PARTS = ["default-src 'self'", "script-src 'self'", "style-src 'self'", "img-src 'self' data:", "connect-src 'self'",
             "media-src 'self' blob:", "worker-src 'self'", "object-src 'none'", "base-uri 'none'", "frame-ancestors 'none'"]


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(models=models([[]]), settings=Settings())) as c:
        wait_ready(c)
        yield c


def check(r):
    for k, v in REQUIRED.items():
        assert r.headers.get(k) == v, (r.url, k, r.headers.get(k))
    csp = r.headers.get("content-security-policy", "")
    for part in CSP_PARTS:
        assert part in csp, (r.url, part, csp)


@pytest.mark.parametrize("path", ["/", "/health", "/js/main.js", "/css/app.css", "/samples/q_warranty_en.wav", "/does-not-exist"])
def test_headers_on_index_health_static_files_and_404(client, path):
    check(client.get(path))


def test_headers_on_head_and_range_requests(client):
    check(client.head("/"))
    r = client.get("/samples/q_warranty_en.wav", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206
    check(r)


def test_csp_has_no_unsafe_inline_or_eval(client):
    csp = client.get("/").headers["content-security-policy"]
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp and "*" not in csp


def test_index_html_has_no_inline_style_or_script():
    """A strict CSP blocks all of these; keep the page free of them instead of loosening the policy."""
    html = (ROOT / "client" / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"\sstyle\s*=", html), "inline style attribute"
    assert "<style" not in html, "style element"
    assert not re.search(r"<script(?![^>]*\ssrc=)", html), "inline script"
    assert not re.search(r"\son[a-z]+\s*=", html, re.I), "inline event handler"
    assert "javascript:" not in html.lower()
