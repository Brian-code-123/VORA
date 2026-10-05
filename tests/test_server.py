import asyncio
import json
import threading
import time
import wave

import numpy as np
import pytest

from vora.rag import store
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.test_pipeline import HIT, FakeLlm, FakeRetriever, FakeTts, fin, part
from vora.config import Settings
from vora.pipeline import Executors
from vora.server import Models, create_app, serve_session

FRAME = b"\x00\x00" * 1600


class ScriptAsr:
    def __init__(self, script):
        self.script, self.i = script, 0

    def feed(self, pcm):
        ev = self.script[self.i] if self.i < len(self.script) else []
        self.i += 1
        return ev


def models(scripts, tts=None, llm=None):
    it = iter(scripts)
    return Models(lambda lang: ScriptAsr(next(it)), FakeRetriever(), llm or FakeLlm(), tts or FakeTts(), Executors())


def app_with(m, **kw):
    return create_app(models=m, settings=Settings(**kw))


def recv_until(ws, kind="metrics", limit=200):
    got = []
    for _ in range(limit):
        msg = ws.receive()
        if msg.get("bytes") is not None:
            got.append(("audio", msg["bytes"]))
        elif msg.get("text") is not None:
            d = json.loads(msg["text"])
            got.append(("json", d))
            if d["type"] == kind:
                break
    return got


def wait_ready(c):
    for _ in range(100):
        if c.get("/health").status_code == 200:
            return
        time.sleep(0.05)
    raise AssertionError("not ready")


def test_ws_roundtrip_en_question_returns_audio():
    with TestClient(app_with(models([[[fin("hello there")]]]))) as c:
        wait_ready(c)
        with c.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "config", "lang": "en"}))
            ws.send_bytes(FRAME)
            got = recv_until(ws)
    types = [d["type"] for k, d in got if k == "json"]
    assert "final" in types and "context" in types and "metrics" in types
    assert sum(len(b) for k, b in got if k == "audio") > 0


def test_two_concurrent_sessions_independent_transcripts():
    m = models([[[fin("question one")]], [[fin("question two")]]])
    with TestClient(app_with(m)) as c:
        wait_ready(c)
        with c.websocket_connect("/ws") as a, c.websocket_connect("/ws") as b:
            a.send_bytes(FRAME)
            b.send_bytes(FRAME)
            ga, gb = recv_until(a), recv_until(b)
    ta = [d["text"] for k, d in ga if k == "json" and d["type"] == "final"]
    tb = [d["text"] for k, d in gb if k == "json" and d["type"] == "final"]
    assert sorted(ta + tb) == ["question one", "question two"] and ta != tb


def test_health_503_until_warm():
    gate = threading.Event()

    def loader(s):
        gate.wait(5)
        return models([[]])

    with TestClient(create_app(settings=Settings(), loader=loader)) as c:
        assert c.get("/health").status_code == 503
        gate.set()
        wait_ready(c)


# ---- stub-driven tests: full control over the client side -------------------------------------
class StubWs:
    def __init__(self, send_delay=0.0):
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent, self.closed, self.send_delay = [], None, send_delay

    async def accept(self): ...

    async def receive(self):
        return await self.inbox.get()

    async def send_bytes(self, b):
        await asyncio.sleep(self.send_delay)
        self.sent.append(("audio", b))

    async def send_json(self, d):
        self.sent.append(("json", d))

    async def close(self, code=1000, reason=None):
        self.closed, self.reason = code, reason
        await self.inbox.put({"type": "websocket.disconnect"})

    def push(self, b=FRAME):
        self.inbox.put_nowait({"type": "websocket.receive", "bytes": b})


async def test_slow_client_queue_bounded():
    m = models([[[fin("hello")]]], tts=FakeTts(chunks=300, delay=0.001))
    s, sessions, ws = Settings(queue_max=4), set(), StubWs(send_delay=0.05)
    t = asyncio.create_task(serve_session(ws, m, s, sessions))
    await asyncio.sleep(0.05)
    ws.push()
    peak = 0
    for _ in range(30):
        await asyncio.sleep(0.02)
        peak = max([peak] + [p.out.qsize() for p in sessions])
    assert 0 < peak <= 4
    ws.inbox.put_nowait({"type": "websocket.disconnect"})
    await t


async def first_audio_delay(m, s, ws_slow=None):
    sessions = set()
    b = StubWs()
    tb = asyncio.create_task(serve_session(b, m, s, sessions))
    await asyncio.sleep(0.05)
    t0 = time.perf_counter()
    b.push()
    while not any(k == "audio" for k, _ in b.sent):
        await asyncio.sleep(0.005)
    dt = time.perf_counter() - t0
    b.inbox.put_nowait({"type": "websocket.disconnect"})
    await tb
    return dt


async def test_slow_client_does_not_starve_other_session():
    base_m = models([[[fin("hello there")]]], tts=FakeTts(chunks=3, delay=0.02))
    base = await first_audio_delay(base_m, Settings(queue_max=4))
    # A: slow reader, long reply; B: normal, same shared executors/models
    m = models([[[fin("slow one")]], [[fin("fast one")]]], tts=FakeTts(chunks=400, delay=0.001))
    s, sessions = Settings(queue_max=4), set()
    a = StubWs(send_delay=0.2)
    ta = asyncio.create_task(serve_session(a, m, s, sessions))
    await asyncio.sleep(0.05)
    a.push()
    await asyncio.sleep(0.3)  # A's TTS worker is now blocked on its full queue
    dt = await first_audio_delay_shared(m, s, sessions)
    a.inbox.put_nowait({"type": "websocket.disconnect"})
    await ta
    assert dt <= 2 * base + 0.1, (dt, base)


async def first_audio_delay_shared(m, s, sessions):
    b = StubWs()
    tb = asyncio.create_task(serve_session(b, m, s, sessions))
    await asyncio.sleep(0.05)
    t0 = time.perf_counter()
    b.push()
    while not any(k == "audio" for k, _ in b.sent):
        await asyncio.sleep(0.005)
    dt = time.perf_counter() - t0
    b.inbox.put_nowait({"type": "websocket.disconnect"})
    await tb
    return dt


async def test_disconnect_cancels_work():
    tts = FakeTts(chunks=500, delay=0.005)
    m = models([[[fin("hello")]]], tts=tts)
    sessions, ws = set(), StubWs()
    t = asyncio.create_task(serve_session(ws, m, Settings(queue_max=64), sessions))
    await asyncio.sleep(0.05)
    ws.push()
    await asyncio.sleep(0.2)
    ws.inbox.put_nowait({"type": "websocket.disconnect"})
    await t
    await asyncio.sleep(0.1)
    assert sessions == set() and tts.cancelled >= 1


async def test_fifth_session_rejected():
    m = models([[]] * 5)
    s, sessions = Settings(max_sessions=4), set()
    open_ = [StubWs() for _ in range(4)]
    tasks = [asyncio.create_task(serve_session(w, m, s, sessions)) for w in open_]
    await asyncio.sleep(0.05)
    fifth = StubWs()
    await serve_session(fifth, m, s, sessions)
    assert fifth.closed == 1013
    for w in open_:
        w.inbox.put_nowait({"type": "websocket.disconnect"})
    await asyncio.gather(*tasks)


async def test_oversize_frame_closes_1009():
    ws, s = StubWs(), Settings(max_frame_bytes=1000)
    t = asyncio.create_task(serve_session(ws, models([[]]), s, set()))
    await asyncio.sleep(0.02)
    ws.push(b"\x00" * 2000)
    await t
    assert ws.closed == 1009


async def test_idle_timeout_closes():
    ws = StubWs()
    await serve_session(ws, models([[]]), Settings(idle_timeout_s=0.2), set())
    assert ws.closed == 1000


@pytest.mark.skipif(not (Settings().models_dir / "llm").exists() or not store.exists(Settings().index_dir),
                    reason="needs models + index")
def test_real_models_roundtrip_en():
    wav = Settings().models_dir / "asr_en/test_wavs/1.wav"
    with wave.open(str(wav)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    x = np.concatenate([pcm, np.zeros(16000 * 2, dtype=np.int16)])
    with TestClient(create_app(settings=Settings())) as c:
        for _ in range(600):
            if c.get("/health").status_code == 200:
                break
            time.sleep(0.5)
        else:
            raise AssertionError("models never became ready")
        with c.websocket_connect("/ws") as ws:
            for i in range(0, len(x), 1600):
                ws.send_bytes(x[i:i + 1600].tobytes())
            got = recv_until(ws, "metrics", limit=2000)
    types = {d["type"] for k, d in got if k == "json"}
    assert {"partial", "final", "context", "metrics"} <= types
    assert sum(len(b) for k, b in got if k == "audio") > 16000


def test_client_files_served():
    with TestClient(app_with(models([[]]))) as c:
        html = c.get("/")
        assert html.status_code == 200 and 'src="js/main.js"' in html.text
        assert "echoCancellation" in c.get("/js/audio.js").text and "worklet.js" in c.get("/js/audio.js").text
        js = c.get("/worklet.js")
        assert js.status_code == 200 and "registerProcessor" in js.text


def test_cross_origin_websocket_rejected():
    with TestClient(app_with(models([[]] * 2))) as c:
        wait_ready(c)
        with pytest.raises(WebSocketDisconnect) as e:
            with c.websocket_connect("/ws", headers={"origin": "http://evil.example"}) as ws:
                ws.receive()
        assert e.value.code == 1008
        with c.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:   # same origin as Host
            ws.send_text(json.dumps({"type": "stop"}))


def test_load_failure_is_reported_on_health():
    def loader(s):
        raise RuntimeError("boom: no int8 voice")

    with TestClient(create_app(settings=Settings(), loader=loader)) as c:
        for _ in range(50):
            r = c.get("/health")
            if r.json().get("error"):
                break
            time.sleep(0.05)
        assert r.status_code == 503 and "boom" in r.json()["error"]


async def test_oversize_text_frame_closes_1009_and_non_object_json_ignored():
    ws, s = StubWs(), Settings(max_text_frame_bytes=200)
    m = models([[[fin("hello there")]]])
    t = asyncio.create_task(serve_session(ws, m, s, set()))
    await asyncio.sleep(0.02)
    ws.inbox.put_nowait({"type": "websocket.receive", "text": "[1, 2]"})      # not an object: ignored
    ws.push()
    await asyncio.sleep(0.5)
    assert any(k == "json" and d["type"] == "final" for k, d in ws.sent)
    ws.inbox.put_nowait({"type": "websocket.receive", "text": "x" * 500})
    await t
    assert ws.closed == 1009


async def test_stalled_client_session_is_closed_1011():
    m = models([[[fin("q one")], [fin("q two")], [fin("q three")], [fin("q four")]]], tts=FakeTts(chunks=200, delay=0.001))
    ws, s, sessions = StubWs(send_delay=999), Settings(queue_max=1, stall_timeout_s=0.2), set()
    t = asyncio.create_task(serve_session(ws, m, s, sessions))
    await asyncio.sleep(0.02)
    for _ in range(4):
        ws.push()
        await asyncio.sleep(0.4)
    await asyncio.wait_for(t, 5)
    assert ws.closed == 1011 and sessions == set()


async def test_close_reasons_present():
    """Every close the server initiates carries a short reason, so a client (or a person reading devtools) can tell why."""
    from vora.server import CLOSE_REASONS
    assert {1000, 1008, 1009, 1011, 1013} <= set(CLOSE_REASONS) and all(0 < len(r.encode()) <= 123 for r in CLOSE_REASONS.values())
    m = models([[]])
    ws = StubWs()
    t = asyncio.create_task(serve_session(ws, m, Settings(max_frame_bytes=10), set()))
    await asyncio.sleep(0.02)
    ws.push(b"x" * 20)
    await asyncio.wait_for(t, 2)
    assert ws.closed == 1009 and ws.reason == CLOSE_REASONS[1009]


def test_health_reports_sessions_and_limit():
    loader = lambda s: models([[]])     # noqa: E731
    with TestClient(create_app(settings=Settings(max_sessions=3), loader=loader)) as c:
        wait_ready(c)
        body = c.get("/health").json()
        assert body["ready"] is True and body["sessions"] == 0 and body["max_sessions"] == 3


def test_client_files_revalidate():
    loader = lambda s: models([[]])     # noqa: E731
    with TestClient(create_app(settings=Settings(), loader=loader)) as c:
        r = c.get("/js/main.js")
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache" and "javascript" in r.headers["content-type"]
