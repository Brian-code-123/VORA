"""The access key must reach no log. uvicorn writes the request path WITH its query string into the access log
(`WebSocket /ws?key=...`), so a key in a URL ends up in `docker logs`. Checked against a real uvicorn server."""
import asyncio
import json
import logging
import socket
import threading
import time
import urllib.request

import uvicorn
import websockets

from tests.test_pipeline import fin
from tests.test_server import FRAME, models
from vora.config import Settings
from vora.server import create_app

KEY = "s3cret-key-xyz"
WRONG = "wrong-key-abc"


class Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_key_never_appears_in_any_log_record():
    port = free_port()
    app = create_app(models=models([[[fin("hello there")]]]), settings=Settings(access_key=KEY, auth_timeout_s=5.0))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info"))
    seen = Collect()
    names = ("uvicorn", "uvicorn.error", "uvicorn.access", "vora")
    for n in names:
        logging.getLogger(n).addHandler(seen)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started
        await asyncio.to_thread(urllib.request.urlopen, f"http://127.0.0.1:{port}/health")
        await asyncio.to_thread(urllib.request.urlopen, f"http://127.0.0.1:{port}/")

        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:          # wrong key
            await ws.send(json.dumps({"type": "auth", "key": WRONG}))
            try:
                await asyncio.wait_for(ws.recv(), 5)
            except websockets.exceptions.ConnectionClosed as e:
                assert e.rcvd.code == 1008 and "key" in e.rcvd.reason
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:          # right key: a real turn must be served
            await ws.send(json.dumps({"type": "auth", "key": KEY}))
            await ws.send(json.dumps({"type": "config", "lang": "en"}))
            await ws.send(FRAME)
            got = []
            for _ in range(50):
                m = await asyncio.wait_for(ws.recv(), 10)
                if isinstance(m, str):
                    got.append(json.loads(m)["type"])
                    if got[-1] == "final":
                        break
            assert "final" in got, got
    finally:
        server.should_exit = True
        thread.join(10)
        for n in names:
            logging.getLogger(n).removeHandler(seen)

    text = "\n".join(seen.lines)
    assert KEY not in text and WRONG not in text, text
    assert "key=" not in text, text
    assert any('WebSocket /ws"' in line for line in seen.lines), seen.lines      # the request is logged, without a query
