"""Access-key handshake for /ws. The key travels in the first message, never in a URL: uvicorn logs request paths with
their query strings, so a `?key=` would end up in `docker logs`."""
import asyncio
import hmac
import json
import logging
import time
from collections import OrderedDict, deque
from typing import Callable

from vora.config import Settings

log = logging.getLogger("vora")
KEY_REASON = "access key missing or wrong"
THROTTLE_REASON = "too many failed attempts, wait a minute"      # must not mention the key: the page maps that wording to the wrong-key hint


class AuthThrottle:
    """Failed handshakes per address in a sliding window. 96 bits of key make guessing hopeless; this stops a stranger
    from hammering the port (each attempt costs a socket and a log line). In memory, bounded: the least recently active
    address is dropped beyond max_ips. Attempts refused while blocked are not recorded, so the block cannot be extended
    by hammering and always ends window_s after the last real failure."""

    def __init__(self, max_fails: int = 5, window_s: float = 60.0, max_ips: int = 10_000,
                 clock: Callable[[], float] = time.monotonic):
        self.max_fails, self.window_s, self.max_ips, self.clock = max_fails, window_s, max_ips, clock
        self._hits: OrderedDict[str, deque] = OrderedDict()

    def __len__(self) -> int:
        return len(self._hits)

    def _recent(self, ip: str | None) -> deque:
        q = self._hits.get(ip or "?")
        if q is None:
            return deque()
        cutoff = self.clock() - self.window_s
        while q and q[0] <= cutoff:
            q.popleft()
        return q

    def blocked(self, ip: str | None) -> bool:
        return len(self._recent(ip)) >= self.max_fails

    def fail(self, ip: str | None) -> None:
        key = ip or "?"
        q = self._recent(key)
        q.append(self.clock())
        self._hits[key] = q
        self._hits.move_to_end(key)
        while len(self._hits) > self.max_ips:
            self._hits.popitem(last=False)


async def handshake(ws, s: Settings, throttle: AuthThrottle | None = None) -> bool:
    """`ws` must already be accepted. True = serve the session. False = the socket has been closed 1008.
    Without a configured key (localhost use) nothing is read and any `auth` message later on is ignored."""
    if not s.access_key:
        return True
    ip = _ip(ws)
    if throttle is not None and throttle.blocked(ip):
        await ws.close(code=1008, reason=THROTTLE_REASON)          # refused before reading, and not recorded
        return False
    try:
        msg = await asyncio.wait_for(ws.receive(), s.auth_timeout_s)
    except asyncio.TimeoutError:
        msg = {}
    if _key_ok(msg, s.access_key):
        return True
    if throttle is not None:
        throttle.fail(ip)
    log.warning("ws auth failed from %s", ip)                      # the address only, never the key that was offered
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
