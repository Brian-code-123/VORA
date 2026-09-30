import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from vora.config import ROOT, Settings
from vora.pipeline import Executors, Pipeline

log = logging.getLogger("vora")


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
        m = cls(lambda lang: AsrSession(recs, lang), Retriever(s), Llm(s), Tts(s), Executors())
        # warmup: first real call pays page-in / graph init (measured 3-8 s for the LLM)
        for lang in ("en", "zh"):
            m.make_asr(lang).feed(np.zeros(1600, dtype=np.int16).tobytes())
        m.retriever.search("warm up")
        list(m.llm.stream("hi", [Hit("w", "Hello.", 1.0)]))
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
        await ws.close(code=1013)
        return
    pipe = Pipeline(models.make_asr("en"), models.retriever, models.llm, models.tts, s, models.executors)
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
                    await ws.close(code=1009)
                    break
                await pipe.on_audio(msg["bytes"])
            elif msg.get("text"):
                try:
                    d = json.loads(msg["text"])
                except ValueError:
                    continue
                if d.get("type") == "stop":
                    break
                if d.get("type") == "config" and d.get("lang") in ("en", "zh") and d["lang"] != lang:
                    lang = d["lang"]
                    await pipe.close()
                    pipe.asr = models.make_asr(lang)
    except asyncio.TimeoutError:
        await ws.close(code=1000)
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        await pipe.close()
        sessions.discard(pipe)


def create_app(models: Models | None = None, settings: Settings | None = None,
               loader: Callable[[Settings], Models] | None = None) -> FastAPI:
    s = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready, app.state.models, app.state.sessions = False, models, set()

        async def load():
            app.state.models = await asyncio.to_thread(loader or Models.load, s) if models is None else models
            app.state.ready = True

        task = asyncio.create_task(load())  # not awaited: /health answers 503 while models warm up
        yield
        task.cancel()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return JSONResponse({"ready": app.state.ready}, status_code=200 if app.state.ready else 503)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        if not app.state.ready:
            await ws.close(code=1013)
            return
        await serve_session(ws, app.state.models, s, app.state.sessions)

    client_dir = ROOT / "client"
    if client_dir.exists():
        app.mount("/", StaticFiles(directory=client_dir, html=True), name="client")
    return app


def main() -> None:
    import uvicorn
    s = Settings()
    uvicorn.run(create_app(settings=s), host=s.host, port=8000, log_level="info")


if __name__ == "__main__":
    main()
