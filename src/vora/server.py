import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Callable

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from vora.config import ROOT, Settings
from vora.pipeline import Admission, ClientStalled, Executors, Pipeline

log = logging.getLogger("vora")
CLOSE_REASONS = {1000: "idle timeout", 1008: "origin not allowed", 1009: "frame too large", 1011: "client stopped reading",
                 1013: "busy or loading"}


async def _close(ws, code: int) -> None:
    await ws.close(code=code, reason=CLOSE_REASONS.get(code, ""))


@dataclass
class Models:
    make_asr: Callable[[str], Any]
    retriever: Any
    llm: Any
    tts: Any
    executors: Executors

    @classmethod
    def load(cls, s: Settings) -> "Models":
        from vora.asr import AsrSession, load_recognizers
        from vora.llm import Llm
        from vora.rag.retriever import Retriever
        from vora.rag.store import Hit
        from vora.tts import Tts

        recs = load_recognizers(s)
        m = cls(lambda lang: AsrSession(recs, lang, hold_ms=s.hold_ms, hold_total_cap_ms=s.hold_total_cap_ms), Retriever(s), Llm(s), Tts(s), Executors(admission=Admission(s.max_inflight_turns)))
        # warmup: first real call pays page-in / graph init (measured 3-8 s for the LLM)
        for lang in ("en", "zh"):
            m.make_asr(lang).feed(np.zeros(1600, dtype=np.int16).tobytes())
        m.retriever.search("warm up")
        m.retriever.search("你好，保修期多久")    # loads the zh embedder now, not on the first Chinese question (measured 6 s)
        list(m.llm.stream("hi", [Hit("w", "Hello.", 1.0)]))
        list(m.llm.stream("你好", [Hit("w", "你好。", 1.0)]))  # pages in CJK embedding rows (cold zh call took 4.7 s)
        list(m.tts.synth("Hello.")), list(m.tts.synth("你好。"))
        return m


async def _send_loop(ws, pipe: Pipeline) -> None:
    try:
        while True:
            kind, payload = await pipe.out.get()
            if kind == "audio":
                await ws.send_bytes(payload)
            else:
                if payload["type"] == "metrics":
                    log.info(json.dumps({"event": "turn", **payload}))
                await ws.send_json(payload)
    except Exception:  # client gone; the receive loop notices and cleans up
        return


async def serve_session(ws, models: Models, s: Settings, sessions: set) -> None:
    """Duck-typed on `ws` (accept/receive/send_json/send_bytes/close) so tests can drive it with a stub."""
    await ws.accept()
    if len(sessions) >= s.max_sessions:
        await _close(ws, 1013)
        return
    pipe = Pipeline(models.make_asr("en"), models.retriever, models.llm, models.tts, s, models.executors,
                    active_sessions=lambda: len(sessions))
    sessions.add(pipe)
    sender = asyncio.create_task(_send_loop(ws, pipe))
    lang = "en"
    try:
        while True:
            msg = await asyncio.wait_for(ws.receive(), s.idle_timeout_s)
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if len(msg["bytes"]) > s.max_frame_bytes:
                    await _close(ws, 1009)
                    break
                await pipe.on_audio(msg["bytes"])
            elif msg.get("text"):
                if len(msg["text"]) > s.max_text_frame_bytes:
                    await _close(ws, 1009)
                    break
                try:
                    d = json.loads(msg["text"])
                except ValueError:
                    continue
                if not isinstance(d, dict):
                    continue
                if d.get("type") == "stop":
                    break
                if d.get("type") == "config" and d.get("lang") in ("en", "zh") and d["lang"] != lang:
                    lang = d["lang"]
                    await pipe.close()
                    pipe.asr = models.make_asr(lang)
    except asyncio.TimeoutError:
        await _close(ws, 1000)
    except ClientStalled:  # client keeps the socket open but never reads: free the slot
        await _close(ws, 1011)
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        await pipe.close()
        sessions.discard(pipe)


class ClientFiles(StaticFiles):
    """The client is a handful of small ES modules: revalidate on every load (ETag) so an upgrade never runs stale JS."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


def create_app(models: Models | None = None, settings: Settings | None = None,
               loader: Callable[[Settings], Models] | None = None) -> FastAPI:
    s = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready, app.state.models, app.state.sessions, app.state.error = False, models, set(), None

        async def load():
            try:
                app.state.models = await asyncio.to_thread(loader or Models.load, s) if models is None else models
                app.state.ready = True
            except Exception as e:  # noqa: BLE001 - surfaced on /health and in the log, never a silent 503 forever
                log.exception("model load failed")
                app.state.error = f"{type(e).__name__}: {e}"

        task = asyncio.create_task(load())  # not awaited: /health answers 503 while models warm up
        yield
        task.cancel()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        body = {"ready": app.state.ready, "sessions": len(app.state.sessions), "max_sessions": s.max_sessions}
        if app.state.error:
            body["error"] = app.state.error
        return JSONResponse(body, status_code=200 if app.state.ready else 503)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        origin = ws.headers.get("origin")
        if origin and urlparse(origin).netloc != ws.headers.get("host") and origin not in s.allowed_origins:
            await _close(ws, 1008)   # any web page could otherwise drive a localhost server from the user's browser
            return
        if not app.state.ready:
            await _close(ws, 1013)
            return
        await serve_session(ws, app.state.models, s, app.state.sessions)

    client_dir = ROOT / "client"
    if client_dir.exists():
        app.mount("/", ClientFiles(directory=client_dir, html=True), name="client")
    return app


def main() -> None:
    import uvicorn
    s = Settings()
    kw = {"ssl_certfile": s.ssl_cert, "ssl_keyfile": s.ssl_key} if s.ssl_cert else {}
    uvicorn.run(create_app(settings=s), host=s.host, port=8000, log_level="info", **kw)


if __name__ == "__main__":
    main()
