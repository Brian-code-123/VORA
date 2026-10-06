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
