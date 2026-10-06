"""Access-key handshake for /ws. The key travels in the first message, never in a URL: uvicorn logs request paths with
their query strings, so a `?key=` would end up in `docker logs`."""
import asyncio
import hmac
import json
import logging

from vora.config import Settings

log = logging.getLogger("vora")
KEY_REASON = "access key missing or wrong"


async def handshake(ws, s: Settings) -> bool:
    """`ws` must already be accepted. True = serve the session. False = the socket has been closed 1008.
    Without a configured key (localhost use) nothing is read and any `auth` message later on is ignored."""
    if not s.access_key:
        return True
    try:
        msg = await asyncio.wait_for(ws.receive(), s.auth_timeout_s)
    except asyncio.TimeoutError:
        msg = {}
    if _key_ok(msg, s.access_key):
        return True
    log.warning("ws auth failed from %s", _ip(ws))      # the address only, never the key that was offered
    await ws.close(code=1008, reason=KEY_REASON)
    return False


def _key_ok(msg: dict, expected: str) -> bool:
    raw = msg.get("text")
    if not isinstance(raw, str):
        return False
    try:
        d = json.loads(raw)
    except ValueError:
        return False
    if not isinstance(d, dict) or d.get("type") != "auth" or not isinstance(d.get("key"), str):
        return False
    return hmac.compare_digest(d["key"].encode(), expected.encode())


def _ip(ws) -> str:
    c = getattr(ws, "client", None)
    return getattr(c, "host", None) or "?"
