"""WebSocket access-key handshake: the key travels in the first message, never in a URL (URLs end up in access logs)."""
import asyncio
import json
import threading

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.test_pipeline import fin
from tests.test_server import FRAME, models, wait_ready
from vora.auth import handshake
from vora.config import Settings
from vora.server import create_app

KEY = "s3cret-key"


class AuthWs:
    """Accepted socket stub: hands out the given frames, then stays silent."""

    def __init__(self, *frames):
        self.frames, self.closed, self.reason, self.reads = list(frames), None, None, 0

    async def receive(self):
        self.reads += 1
        if self.frames:
            return self.frames.pop(0)
        await asyncio.sleep(10)

    async def close(self, code=1000, reason=None):
        self.closed, self.reason = code, reason


def text(d):
    return {"type": "websocket.receive", "text": d if isinstance(d, str) else json.dumps(d)}


def auth(key):
    return text({"type": "auth", "key": key})


def served(ws) -> tuple[list, int | None]:
    """Drive a session until its metrics arrive or the server closes it: (event types, close code or None).
    A refused or busy socket ends with the close code; an authenticated one yields final/context/metrics events."""
    types = []
    for _ in range(200):
        msg = ws.receive()
        if msg["type"] == "websocket.close":
            return types, msg["code"]
        if msg.get("text") is not None:
            types.append(json.loads(msg["text"])["type"])
            if types[-1] == "metrics":
                return types, None
    raise AssertionError(types)


def one_turn_models():
    return models([[[fin("hello there")]]])


def S(key=KEY, **kw):
    return Settings(access_key=key, auth_timeout_s=0.05, **kw)


async def test_handshake_accepts_correct_key():
    ws = AuthWs(auth(KEY))
    assert await handshake(ws, S()) is True and ws.closed is None


async def test_wrong_key_closes_1008_with_key_reason():
    ws = AuthWs(auth("nope"))
    assert await handshake(ws, S()) is False
    assert ws.closed == 1008 and "key" in ws.reason


async def test_no_frame_within_timeout_closes():
    ws = AuthWs()
    assert await handshake(ws, S()) is False
    assert ws.closed == 1008 and "key" in ws.reason


async def test_binary_first_frame_rejected():
    ws = AuthWs({"type": "websocket.receive", "bytes": b"\x00\x00"})
    assert await handshake(ws, S()) is False and ws.closed == 1008


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", "3", "null", '"auth"', "{"])
async def test_non_json_and_non_object_first_frame_rejected(raw):
    ws = AuthWs(text(raw))
    assert await handshake(ws, S()) is False and ws.closed == 1008


@pytest.mark.parametrize("bad", [None, 123, ["k"], {"k": 1}, True])
async def test_key_not_a_string_rejected_without_exception(bad):
    ws = AuthWs(auth(bad))
    assert await handshake(ws, S()) is False and ws.closed == 1008


async def test_missing_key_field_rejected():
    ws = AuthWs(text({"type": "auth"}))
    assert await handshake(ws, S()) is False and ws.closed == 1008


async def test_empty_and_unicode_key():
    assert await handshake(AuthWs(auth("")), S()) is False
    uni = "密钥-ключ-🔑"
    assert await handshake(AuthWs(auth(uni)), S(key=uni)) is True
    assert await handshake(AuthWs(auth("密钥")), S(key=uni)) is False


async def test_config_before_auth_counts_as_failed_auth():
    """Clients from before this change send `config` first: they must be refused, not served unauthenticated."""
    ws = AuthWs(text({"type": "config", "lang": "en"}), auth(KEY))
    assert await handshake(ws, S()) is False and ws.closed == 1008


async def test_second_auth_message_is_ignored_after_success():
    ws = AuthWs(auth(KEY), auth("other"))
    assert await handshake(ws, S()) is True
    assert ws.reads == 1 and ws.closed is None       # only the first frame is consumed


async def test_keyless_server_never_waits_and_leaves_frames_alone():
    ws = AuthWs(auth("whatever"))
    assert await handshake(ws, Settings(access_key="")) is True
    assert ws.reads == 0 and ws.closed is None


def test_keyless_server_ignores_auth_message():
    with TestClient(create_app(models=one_turn_models(), settings=Settings())) as c:
        wait_ready(c)
        with c.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "auth", "key": "x"}))
            ws.send_bytes(FRAME)
            types, code = served(ws)
    assert code is None and "final" in types and "metrics" in types


def test_server_accepts_first_message_key_and_refuses_query_key():
    with TestClient(create_app(models=one_turn_models(), settings=Settings(access_key=KEY))) as c:
        wait_ready(c)
        with pytest.raises(WebSocketDisconnect) as e:               # a key in the URL is no longer a credential
            with c.websocket_connect(f"/ws?key={KEY}") as ws:
                ws.receive_text()
        assert e.value.code == 1008 and "key" in (e.value.reason or "")
        with c.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "auth", "key": KEY}))
            ws.send_text(json.dumps({"type": "config", "lang": "en"}))
            ws.send_bytes(FRAME)
            types, code = served(ws)
    assert code is None and "final" in types and "context" in types and "metrics" in types


def test_unauthenticated_socket_does_not_take_a_session_slot():
    """A flood of sockets that never authenticate must not starve real users (max_sessions counts authenticated ones)."""
    s = Settings(access_key=KEY, max_sessions=1, auth_timeout_s=30.0)
    with TestClient(create_app(models=one_turn_models(), settings=s)) as c:
        wait_ready(c)
        with c.websocket_connect("/ws"):                             # idle, never authenticates
            with c.websocket_connect("/ws") as real:
                real.send_text(json.dumps({"type": "auth", "key": KEY}))
                real.send_bytes(FRAME)
                types, code = served(real)
    assert code is None and "final" in types                         # served, not closed 1013 "busy"


def test_ready_check_happens_after_auth():
    """While models load, a wrong key still gets the key refusal (no readiness oracle for strangers); the right key gets busy."""
    go = threading.Event()
    m = models([[]])
    app = create_app(models=None, settings=Settings(access_key=KEY, auth_timeout_s=5.0), loader=lambda s: (go.wait(30), m)[1])
    with TestClient(app) as c:
        try:
            assert c.get("/health").status_code == 503
            with c.websocket_connect("/ws") as ws:
                ws.send_text(json.dumps({"type": "auth", "key": "wrong"}))
                msg = ws.receive()
                assert msg["code"] == 1008 and "key" in msg["reason"]
            with c.websocket_connect("/ws") as ws:
                ws.send_text(json.dumps({"type": "auth", "key": KEY}))
                assert ws.receive()["code"] == 1013
        finally:
            go.set()


@pytest.mark.parametrize("key,expected", [("K", ["auth", "config"]), ("", ["config"])])
async def test_ws_smoke_sends_auth_first_when_given_a_key(key, expected):
    """The deploy self-test and the owner's checks use scripts/ws_smoke.py: with a key it must authenticate before anything else."""
    from pathlib import Path

    import websockets

    from scripts import ws_smoke
    seen = []

    async def handler(ws):
        for _ in expected:
            seen.append(json.loads(await ws.recv()))
        await ws.close()

    wav = Path(__file__).resolve().parent.parent / "client" / "samples" / "q_warranty_en.wav"
    async with websockets.serve(handler, "127.0.0.1", 0) as srv:
        port = srv.sockets[0].getsockname()[1]
        try:
            await ws_smoke.run(f"ws://127.0.0.1:{port}/ws", str(wav), "en", 2.0, False, False, key=key)
        except websockets.exceptions.ConnectionClosed:
            pass
    assert [m["type"] for m in seen] == expected
    assert key == "" or seen[0]["key"] == key


# ---- throttling of failed attempts (per address, sliding window) ------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from vora.auth import THROTTLE_REASON, AuthThrottle  # noqa: E402


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def thr(clock=None, **kw):
    return AuthThrottle(clock=clock or Clock(), **kw)


def test_four_failures_do_not_block_fifth_does():
    t = thr()
    for _ in range(4):
        t.fail("1.2.3.4")
    assert not t.blocked("1.2.3.4")
    t.fail("1.2.3.4")
    assert t.blocked("1.2.3.4")


def test_block_expires_after_window():
    c = Clock()
    t = thr(c)
    for _ in range(5):
        t.fail("a")
    c.now = 59.9
    assert t.blocked("a")
    c.now = 60.0
    assert not t.blocked("a")


def test_window_is_sliding_not_fixed():
    c = Clock()
    t = thr(c)
    for ts in (0, 10, 20, 30, 40):
        c.now = ts
        t.fail("a")
    assert t.blocked("a")
    c.now = 60.5                       # the failure at t=0 has left the window: 4 remain
    assert not t.blocked("a")
    t.fail("a")                        # 10, 20, 30, 40, 60.5 -> 5 again
    assert t.blocked("a")
    c.now = 70.5                       # 10 has left: 4 remain
    assert not t.blocked("a")


def test_unknown_address_shares_one_bucket():
    t = thr()
    for _ in range(5):
        t.fail(None)
    assert t.blocked(None) and t.blocked("?") and not t.blocked("1.1.1.1")


def test_ipv6_and_ipv4_tracked_separately():
    t = thr()
    for _ in range(5):
        t.fail("::1")
    assert t.blocked("::1") and not t.blocked("127.0.0.1")


def test_memory_capped_at_max_ips_oldest_evicted():
    c = Clock()
    t = thr(c, max_ips=3)
    for _ in range(4):
        t.fail("a")                    # one short of blocked, but the least recently active
    for i, ip in enumerate(("b", "c", "d"), start=1):
        c.now = i
        t.fail(ip)
    assert len(t) <= 3
    c.now = 5
    t.fail("a")                        # evicted earlier: counts from 1, not 5
    assert not t.blocked("a")


def with_client(ws, host):
    ws.client = SimpleNamespace(host=host)
    return ws


async def test_blocked_address_is_refused_before_reading_even_with_the_right_key():
    t = thr()
    for _ in range(5):
        t.fail("9.9.9.9")
    ws = with_client(AuthWs(auth(KEY)), "9.9.9.9")
    assert await handshake(ws, S(), t) is False
    assert ws.closed == 1008 and ws.reason == THROTTLE_REASON and ws.reads == 0


async def test_failed_handshakes_are_counted_wrong_key_timeout_and_garbage():
    t = thr(max_fails=3)
    for frames in ([auth("nope")], [], [text("garbage")]):
        assert await handshake(with_client(AuthWs(*frames), "7.7.7.7"), S(), t) is False
    assert t.blocked("7.7.7.7")


async def test_success_does_not_clear_earlier_failures():
    t = thr(max_fails=2)
    assert await handshake(with_client(AuthWs(auth("x")), "5.5.5.5"), S(), t) is False
    assert await handshake(with_client(AuthWs(auth(KEY)), "5.5.5.5"), S(), t) is True
    assert await handshake(with_client(AuthWs(auth("y")), "5.5.5.5"), S(), t) is False        # second failure
    assert await handshake(with_client(AuthWs(auth(KEY)), "5.5.5.5"), S(), t) is False        # now blocked


async def test_blocked_attempts_do_not_extend_the_block():
    c = Clock()
    t = thr(c, max_fails=2)
    for _ in range(2):
        await handshake(with_client(AuthWs(auth("x")), "4.4.4.4"), S(), t)
    for _ in range(10):                # hammering while blocked
        c.now += 1
        await handshake(with_client(AuthWs(auth("x")), "4.4.4.4"), S(), t)
    c.now = 61
    assert not t.blocked("4.4.4.4")    # the two real failures are 60 s old; the refused attempts were not recorded


def test_throttle_reason_has_no_key_word():
    """The page maps a 1008 whose reason mentions a key to the wrong-key hint; a throttle needs its own message."""
    assert "key" not in THROTTLE_REASON.lower() and "attempts" in THROTTLE_REASON


def test_server_throttles_repeated_wrong_keys():
    s = Settings(access_key=KEY, auth_max_fails=3, auth_window_s=60.0)
    with TestClient(create_app(models=one_turn_models(), settings=s)) as c:
        wait_ready(c)
        for _ in range(3):
            with c.websocket_connect("/ws") as ws:
                ws.send_text(json.dumps({"type": "auth", "key": "wrong"}))
                assert ws.receive()["reason"] == "access key missing or wrong"
        with c.websocket_connect("/ws") as ws:                      # the right key from the same address is refused too
            ws.send_text(json.dumps({"type": "auth", "key": KEY}))
            msg = ws.receive()
            assert msg["code"] == 1008 and msg["reason"] == THROTTLE_REASON
