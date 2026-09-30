"""Per-connection orchestrator: ASR events -> (speculative) retrieval -> LLM tokens -> clauses -> TTS audio.

Threads: ASR/RAG/LLM/TTS each have their own executor. The LLM worker only fills a sentence queue, so the
LLM lock is never held while waiting on a slow client; only the TTS worker blocks on the bounded out queue.
"""
import asyncio
import difflib
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from vora.chunker import SentenceChunker
from vora.config import Settings
from vora.metrics import LatencyTrace

OutMsg = tuple[Literal["json", "audio"], Any]
FILLER = {"zh": "好的，", "en": "Sure, "}
PREFETCH_MIN_CHARS = 6
PREFETCH_INTERVAL_S = 0.5
BARGE_IN_CHUNKS = 2


@dataclass
class Executors:
    asr: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(2, "asr"))
    rag: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(1, "rag"))
    llm: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(1, "llm"))
    tts: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(1, "tts"))


def _lang(text: str) -> str:
    return "zh" if any("一" <= c <= "鿿" for c in text) else "en"


class Pipeline:
    def __init__(self, asr, retriever, llm, tts, settings: Settings, executors: Executors | None = None,
                 clock: Callable[[], float] = time.perf_counter):
        self.asr, self.retriever, self.llm, self.tts, self.s = asr, retriever, llm, tts, settings
        self.ex = executors or Executors()
        self.clock = clock
        self.out: asyncio.Queue[OutMsg] = asyncio.Queue(maxsize=settings.queue_max)
        self.loop = asyncio.get_running_loop()
        self._turn: asyncio.Task | None = None
        self._turn_ev = threading.Event()
        self._partial_run = 0
        self._last_partial = ""
        self._last_voice = clock()
        self._prefetch_text = ""
        self._prefetch_task: asyncio.Task | None = None
        self._prefetch_at = -1e9

    # ---- output -------------------------------------------------------------------------------
    async def _emit(self, msg: dict, droppable: bool = False) -> None:
        if droppable and self.out.full():
            return
        await self.out.put(("json", msg))

    async def _put_audio(self, chunk: bytes, ev: threading.Event) -> None:
        while not ev.is_set():  # poll instead of a blocked put(): a stale chunk must never land after a cancel
            try:
                self.out.put_nowait(("audio", chunk))
                return
            except asyncio.QueueFull:
                await asyncio.sleep(0.005)

    def _drain_audio(self) -> None:
        keep = []
        while not self.out.empty():
            m = self.out.get_nowait()
            if m[0] == "json" and m[1]["type"] not in ("token", "partial"):  # stale audio/display chunks go
                keep.append(m)
        for m in keep:
            self.out.put_nowait(m)

    # ---- input --------------------------------------------------------------------------------
    async def on_audio(self, pcm: bytes) -> None:
        events = await self.loop.run_in_executor(self.ex.asr, self.asr.feed, pcm)
        speaking = False
        final_seen = False
        for e in events:
            if e.kind == "partial":
                if not e.stable:
                    speaking = True
                    self._last_voice = self.clock()
                self._last_partial = e.text
                await self._emit({"type": "partial", "text": e.text, "stable": e.stable}, droppable=True)
                if e.stable and len(e.text) >= PREFETCH_MIN_CHARS:
                    self._maybe_prefetch(e.text)
            else:
                final_seen = True
                await self._on_final(e.text)
        self._partial_run = self._partial_run + 1 if speaking and not final_seen else 0
        if self._turn_active() and self._partial_run >= BARGE_IN_CHUNKS:
            await self._cancel_turn()  # user talks over the reply (echo is removed client-side)

    def _turn_active(self) -> bool:
        return self._turn is not None and not self._turn.done()

    async def wait_idle(self) -> None:
        if self._turn is not None:
            try:
                await self._turn
            except asyncio.CancelledError:
                pass

    async def close(self) -> None:
        await self._cancel_turn(announce=False)

    # ---- speculative retrieval ----------------------------------------------------------------
    def _maybe_prefetch(self, text: str) -> None:
        now = self.clock()
        if text == self._prefetch_text or now - self._prefetch_at < PREFETCH_INTERVAL_S:
            return
        self._prefetch_at, self._prefetch_text = now, text
        self._prefetch_task = self.loop.create_task(self._search(text))

    async def _search(self, text: str):
        return await self.loop.run_in_executor(self.ex.rag, self.retriever.search, text)

    async def _hits_for(self, text: str):
        if self._prefetch_task is not None and difflib.SequenceMatcher(None, text, self._prefetch_text).ratio() >= 0.7:
            try:
                return await self._prefetch_task
            except Exception:
                pass
        return await self._search(text)

    # ---- turns --------------------------------------------------------------------------------
    async def _on_final(self, text: str) -> None:
        text = text.strip()
        speech_end = self._last_voice
        self._partial_run, self._last_partial = 0, ""
        if not text:
            return
        await self._cancel_turn()
        trace = LatencyTrace(self.clock)
        trace.marks["speech_end"] = speech_end
        trace.mark("final")
        await self._emit({"type": "final", "text": text})
        ev = self._turn_ev = threading.Event()
        self._turn = self.loop.create_task(self._run_turn(text, trace, ev))

    async def _cancel_turn(self, announce: bool = True) -> None:
        if not self._turn_active():
            return
        self._turn_ev.set()
        self.llm.cancel()
        self.tts.cancel()
        self._turn.cancel()
        try:
            await self._turn
        except asyncio.CancelledError:
            pass
        self._drain_audio()
        if announce:  # never block on a full queue: the client must learn about the cancel
            while self.out.full():
                self.out.get_nowait()
            self.out.put_nowait(("json", {"type": "cancel"}))

    async def _run_turn(self, text: str, trace: LatencyTrace, ev: threading.Event) -> None:
        hits = await self._hits_for(text)
        await self._emit({"type": "context", "ids": [h.chunk_id for h in hits]})
        sent_q: queue.Queue = queue.Queue()
        stamps: dict[str, float] = {}
        if self.s.filler:
            sent_q.put((FILLER[_lang(text)], True))

        def llm_worker() -> None:
            chunker = SentenceChunker()
            try:
                for tok in self.llm.stream(text, hits):
                    if ev.is_set():
                        return
                    if "first_token" not in trace.marks:
                        trace.mark("first_token")
                    asyncio.run_coroutine_threadsafe(self._emit({"type": "token", "text": tok}, droppable=True), self.loop)
                    for c in chunker.push(tok):
                        sent_q.put((c, False))
                for c in chunker.flush():
                    sent_q.put((c, False))
            finally:
                sent_q.put(None)

        def tts_worker() -> None:
            while not ev.is_set():
                try:
                    item = sent_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is None:
                    return
                sentence, is_filler = item
                for chunk in self.tts.synth(sentence):
                    if ev.is_set():
                        return
                    now = self.clock()
                    stamps.setdefault("first_audio", now)
                    if not is_filler:
                        stamps.setdefault("first_content", now)
                    asyncio.run_coroutine_threadsafe(self._put_audio(chunk, ev), self.loop).result()

        await asyncio.gather(
            self.loop.run_in_executor(self.ex.llm, llm_worker),
            self.loop.run_in_executor(self.ex.tts, tts_worker),
        )
        if ev.is_set():
            return
        first_content = stamps.get("first_content", stamps.get("first_audio"))
        if first_content is not None:
            trace.marks["first_audio"] = first_content  # content audio is what the brief's 1.5 s means
        m = trace.report()
        se = trace.marks["speech_end"]
        if "first_audio" in stamps:
            m["ttfa_ms"] = round((stamps["first_audio"] - se) * 1000, 1)
            m["first_content_audio_ms"] = round((first_content - se) * 1000, 1)
        await self._emit({"type": "metrics", **m})
