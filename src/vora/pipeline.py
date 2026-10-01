"""Per-connection orchestrator: ASR events -> (speculative) retrieval -> LLM tokens -> clauses -> TTS audio.

Threads: ASR/RAG/LLM/TTS each have their own executor. The LLM worker only fills a sentence queue, so the
LLM lock is never held while waiting on a slow client; only the TTS worker blocks on the bounded out queue.
"""
import asyncio
import difflib
import os
import queue
import re
import threading
import time
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

from vora.chunker import SentenceChunker
from vora.config import Settings
from vora.llm import BUSY
from vora.metrics import LatencyTrace

log = logging.getLogger("vora")
OutMsg = tuple[Literal["json", "audio"], Any]
FILLER = {"zh": "好的，", "en": "Sure, "}
PREFETCH_MIN_CHARS = 6
PREFETCH_INTERVAL_S = 0.5
BARGE_IN_CHANGES = 2        # changed partials ...
BARGE_IN_WINDOW_S = 1.5     # ... within this window (real ASR changes text about every 3rd 100 ms frame)
ECHO_WINDOW_S = 10.0
VOICED_MIN_RMS = 250.0


SHADOW_MIN_WORDS = 3
SHADOW_MIN_CJK = 4
SHADOW_PRERENDER_SENTENCES = 2


class ShadowGate:
    """A speculative turn runs fully but is muted: tokens, context, `_spoken` entries and (after 2 sentences) audio are
    held here until the final text matches (promote) or is thrown away (cancel)."""

    def __init__(self):
        self.open = threading.Event()
        self.lock = threading.Lock()
        self.tokens: list[str] = []
        self.spoken: list[tuple[float, str]] = []
        self.audio: list[tuple[bytes, bool]] = []     # (pcm piece, is_filler)
        self.context_ids: list[str] = []
        self.stamps: dict[str, float] = {}

    def take_audio(self) -> list[tuple[bytes, bool]]:
        with self.lock:
            a, self.audio = self.audio, []
            return a


@dataclass
class Shadow:
    text: str
    task: "asyncio.Task"
    ev: threading.Event
    gate: ShadowGate
    trace: "LatencyTrace"


def _same_text(a: str, b: str) -> bool:
    return _alnum(a) == _alnum(b)


class ClientStalled(Exception):
    """The client stopped reading: the bounded output queue stayed full for stall_timeout_s."""


def _alnum(t: str) -> str:
    return re.sub(r"[\W_]+", "", t.lower())


class Admission:
    """Caps in-flight LLM turns (running + queued). Taken BEFORE a turn is submitted to the executor, so a third turn
    behind a busy model gets the 'busy' reply at once instead of waiting unseen in the executor queue."""

    def __init__(self, max_inflight: int):
        self.max = max_inflight
        self._n = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._n >= self.max:
                return False
            self._n += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._n = max(0, self._n - 1)

    @property
    def waiting(self) -> int:
        return self._n


@dataclass
class Executors:
    asr: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(2, "asr"))
    rag: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(1, "rag"))
    llm: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(2, "llm"))  # 2nd user reaches Llm.stream and waits on its lock (3 s -> busy reply)
    tts: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(4, "tts"))  # one blocked-on-client worker per session
    admission: Admission = field(default_factory=lambda: Admission(2))


def _lang(text: str) -> str:
    return "zh" if any("一" <= c <= "鿿" for c in text) else "en"


class Pipeline:
    def __init__(self, asr, retriever, llm, tts, settings: Settings, executors: Executors | None = None,
                 clock: Callable[[], float] = time.perf_counter, active_sessions: Callable[[], int] = lambda: 1):
        self.asr, self.retriever, self.llm, self.tts, self.s = asr, retriever, llm, tts, settings
        self.ex = executors or Executors()
        self.clock = clock
        self.out: asyncio.Queue[OutMsg] = asyncio.Queue(maxsize=settings.queue_max)
        self.loop = asyncio.get_running_loop()
        self._turn: asyncio.Task | None = None
        self._turn_ev = threading.Event()
        self._change_times: deque[float] = deque()
        self._last_partial = ""
        self._last_voice = clock()
        self._floor: float | None = None      # running noise-floor estimate for voiced-frame detection
        self._spoken: deque[tuple[float, str]] = deque(maxlen=64)   # sentences sent to TTS, for echo detection
        self.stalled = False
        self._prefetch_text = ""
        self._prefetch_task: asyncio.Task | None = None
        self._prefetch_at = -1e9
        self._shadow: Shadow | None = None
        self.active_sessions = active_sessions

    # ---- output -------------------------------------------------------------------------------
    async def _emit(self, msg: dict, droppable: bool = False) -> None:
        if droppable and self.out.full():
            return
        try:
            await asyncio.wait_for(self.out.put(("json", msg)), self.s.stall_timeout_s)
        except asyncio.TimeoutError:
            self.stalled = True
            raise ClientStalled() from None

    async def _emit_bg(self, msg: dict, droppable: bool = False) -> None:
        try:
            await self._emit(msg, droppable)
        except ClientStalled:
            pass  # flagged; the receive loop raises on the next frame

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
    def _voiced(self, pcm: bytes) -> bool:
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        self._floor = rms if self._floor is None else min(self._floor * 1.01 + 0.5, rms)
        return rms > max(VOICED_MIN_RMS, 3.0 * self._floor)

    def _is_echo(self, text: str) -> bool:
        """True when `text` is (part of) what we just said: the mic heard our own speaker."""
        now, t = self.clock(), _alnum(text)
        if len(t) < 6:
            return False
        recent = [_alnum(s) for ts, s in self._spoken if now - ts < ECHO_WINDOW_S]
        return any(t in r or r in t or difflib.SequenceMatcher(None, t, r).ratio() >= 0.75 for r in recent if r)

    async def on_audio(self, pcm: bytes) -> None:
        if self.stalled:
            raise ClientStalled()
        now = self.clock()
        if self._voiced(pcm):
            self._last_voice = now   # plan: speech_end = last voiced input frame
        events = await self.loop.run_in_executor(self.ex.asr, self.asr.feed, pcm)
        if self._shadow is not None and self.active_sessions() > 1:
            await self._cancel_shadow()                    # another user arrived: do not burn their CPU on a guess
        for e in events:
            if e.kind == "partial":
                if not e.stable:
                    self._change_times.append(self.clock())
                self._last_partial = e.text
                await self._emit({"type": "partial", "text": e.text, "stable": e.stable}, droppable=True)
                if e.stable and len(e.text) >= PREFETCH_MIN_CHARS:
                    self._maybe_prefetch(e.text)
                if self._shadow is not None and not e.stable and not _same_text(e.text, self._shadow.text):
                    await self._cancel_shadow()            # the user kept talking: the guess is stale
                if e.stable:
                    self._maybe_shadow(e.text)
                if not e.stable and self._turn_active():
                    while self._change_times and self.clock() - self._change_times[0] > BARGE_IN_WINDOW_S:
                        self._change_times.popleft()
                    if len(self._change_times) >= BARGE_IN_CHANGES and not self._is_echo(e.text):
                        await self._cancel_turn()  # user talks over the reply
            else:
                await self._on_final(e.text, getattr(e, "held_ms", 0.0))

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

    async def _hits_for(self, text: str, prefetched: tuple[str, asyncio.Task | None]):
        ptext, ptask = prefetched
        if ptask is not None and ptext == text and not ptask.cancelled():   # exact match only: a prefix must not decide the answer
            try:
                return await asyncio.shield(ptask)   # cancelling this turn must not cancel the shared prefetch
            except asyncio.CancelledError:
                if not ptask.cancelled():
                    raise
            except Exception:
                pass
        return await self._search(text)

    # ---- turns --------------------------------------------------------------------------------
    async def _on_final(self, text: str, held_ms: float = 0.0) -> None:
        text = text.strip()
        speech_end = self._last_voice
        self._change_times.clear()
        self._last_partial = ""
        if not text:
            return
        if self._is_echo(text):
            await self._emit({"type": "echo_ignored", "text": text})
            return
        if self._shadow is not None and _same_text(self._shadow.text, text) and not self._turn_active():
            await self._promote_shadow(text, speech_end, held_ms)
            return
        await self._cancel_shadow()
        await self._cancel_turn()
        prefetched = (self._prefetch_text, self._prefetch_task)
        self._prefetch_text, self._prefetch_task = "", None   # never reuse across utterances
        trace = LatencyTrace(self.clock)
        trace.marks["speech_end"] = speech_end
        trace.mark("final")
        await self._emit({"type": "final", "text": text})
        ev = self._turn_ev = threading.Event()
        self._turn = self.loop.create_task(self._run_turn(text, trace, ev, prefetched, held_ms))

    async def _cancel_turn(self, announce: bool = True) -> None:
        await self._cancel_shadow()
        if not self._turn_active():
            return
        self._turn_ev.set()
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

    # ---- speculative shadow turn --------------------------------------------------------------
    def _shadow_wanted(self, text: str) -> bool:
        if not self.s.speculate or self._shadow is not None or self._turn_active():
            return False
        if self.active_sessions() > 1 or (os.cpu_count() or 1) < self.s.speculate_min_cores:
            return False
        cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        if (cjk < SHADOW_MIN_CJK) if cjk * 2 >= len(text.strip()) else (len(text.split()) < SHADOW_MIN_WORDS):
            return False
        return True

    def _maybe_shadow(self, text: str) -> None:
        if not self._shadow_wanted(text):
            return
        gate, ev = ShadowGate(), threading.Event()
        trace = LatencyTrace(self.clock)
        trace.marks["speech_end"] = self._last_voice
        prefetched = (self._prefetch_text, self._prefetch_task)
        task = self.loop.create_task(self._run_turn(text, trace, ev, prefetched, 0.0, gate))
        self._shadow = Shadow(text, task, ev, gate, trace)

    async def _cancel_shadow(self) -> None:
        sh, self._shadow = self._shadow, None
        if sh is None:
            return
        sh.ev.set()
        sh.task.cancel()
        try:
            await sh.task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 - a dropped guess must never surface
            pass

    async def _promote_shadow(self, text: str, speech_end: float, held_ms: float) -> None:
        """The final text equals the guess: release what the shadow already prepared, in order, immediately."""
        sh, self._shadow = self._shadow, None
        sh.trace.marks["speech_end"] = speech_end
        sh.trace.mark("final")
        await self._emit({"type": "final", "text": text})
        self._turn_ev, self._turn = sh.ev, sh.task
        g = sh.gate
        with g.lock:
            tokens, spoken = list(g.tokens), list(g.spoken)
            g.tokens.clear(), g.spoken.clear()
            g.open.set()
        self._spoken.extend(spoken)
        await self._emit_bg({"type": "context", "ids": g.context_ids})
        for t in tokens:
            await self._emit_bg({"type": "token", "text": t}, droppable=True)
        for chunk, is_filler in g.take_audio():
            now = self.clock()
            g.stamps.setdefault("first_audio", now)
            if not is_filler:
                g.stamps.setdefault("first_content", now)
            await self._put_audio(chunk, sh.ev)

    async def _run_turn(self, text: str, trace: LatencyTrace, ev: threading.Event, prefetched, held_ms: float = 0.0,
                        gate: ShadowGate | None = None) -> None:
        hits = await self._hits_for(text, prefetched)
        if gate is None:
            await self._emit_bg({"type": "context", "ids": [h.chunk_id for h in hits]})
        else:
            gate.context_ids = [h.chunk_id for h in hits]
        sent_q: queue.Queue = queue.Queue()
        stamps: dict[str, float] = gate.stamps if gate is not None else {}
        if self.s.filler:
            self._spoken.append((self.clock(), FILLER[_lang(text)]))
            sent_q.put((FILLER[_lang(text)], True))

        admitted = self.ex.admission.try_acquire()
        if gate is not None and not admitted:
            return                                     # a guess never queues behind (or talks over) a real turn
        released = threading.Event()

        def release_once() -> None:   # exactly one release per acquire, whichever path ends the turn first
            if not released.is_set():
                released.set()
                self.ex.admission.release()

        def record_spoken(sentence: str) -> None:
            if gate is not None:
                with gate.lock:
                    if not gate.open.is_set():
                        gate.spoken.append((self.clock(), sentence))
                        return
            self._spoken.append((self.clock(), sentence))

        def llm_worker() -> None:
            if ev.is_set():
                release_once()
                return
            chunker = SentenceChunker(first_words=self.s.first_chunk_words, second_words=self.s.second_chunk_words)
            try:
                for tok in self.llm.stream(text, hits, ev):
                    if ev.is_set():
                        return
                    if "first_token" not in trace.marks:
                        trace.mark("first_token")
                    if gate is not None:
                        with gate.lock:
                            held = not gate.open.is_set()
                            if held:
                                gate.tokens.append(tok)
                    else:
                        held = False
                    if not held:
                        asyncio.run_coroutine_threadsafe(self._emit_bg({"type": "token", "text": tok}, droppable=True), self.loop)
                    for c in chunker.push(tok):
                        record_spoken(c)
                        sent_q.put((c, False))
                for c in chunker.flush():
                    record_spoken(c)
                    sent_q.put((c, False))
            except Exception:  # noqa: BLE001 - a failing model must not kill the session: log it and say so
                log.exception("LLM turn failed")
                sent_q.put((BUSY[_lang(text)], False))
            finally:
                release_once()
                sent_q.put(None)

        def busy_worker() -> None:
            sent_q.put((BUSY[_lang(text)], False))
            sent_q.put(None)

        rendered = [0]

        def emit_audio(chunk: bytes, is_filler: bool) -> None:
            now = self.clock()
            stamps.setdefault("first_audio", now)
            if not is_filler:
                stamps.setdefault("first_content", now)
            asyncio.run_coroutine_threadsafe(self._put_audio(chunk, ev), self.loop).result()

        def tts_worker() -> None:
            while not ev.is_set():
                try:
                    item = sent_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is None:
                    return
                sentence, is_filler = item
                if gate is not None:
                    # muted until promoted: pre-render only the first sentences, then wait for the final text
                    rendered[0] += 1
                    while not gate.open.is_set() and rendered[0] > SHADOW_PRERENDER_SENTENCES and not ev.is_set():
                        gate.open.wait(0.05)
                    if ev.is_set():
                        return
                for chunk in self.tts.synth(sentence, ev):
                    if ev.is_set():
                        return
                    if gate is not None:
                        with gate.lock:
                            if not gate.open.is_set():
                                gate.audio.append((chunk, is_filler))
                                continue
                        for pc, pf in gate.take_audio():          # anything pre-rendered goes out first
                            emit_audio(pc, pf)
                    emit_audio(chunk, is_filler)

        try:
            await asyncio.gather(
                self.loop.run_in_executor(self.ex.llm, llm_worker) if admitted else self.loop.run_in_executor(None, busy_worker),
                self.loop.run_in_executor(self.ex.tts, tts_worker),
            )
        finally:
            if admitted:
                release_once()   # cancelled before the worker started, or it died: free the slot
        if gate is not None:
            while not gate.open.is_set():              # finished early (short answer): wait to be promoted or cancelled
                await asyncio.sleep(0.01)
        if ev.is_set():
            return
        first_content = stamps.get("first_content", stamps.get("first_audio"))
        if first_content is not None:
            trace.marks["first_audio"] = first_content  # content audio is what the brief's 1.5 s means
        m = trace.report()
        m["tts_oov"] = getattr(self.tts, "oov_total", 0)
        m["hold_ms"] = held_ms
        m["speculated"] = gate is not None
        if gate is not None:
            m["rag_first_token"] = max(0.0, m.get("rag_first_token", 0.0))   # overlapped with the endpoint wait
        se = trace.marks["speech_end"]
        if "first_audio" in stamps:
            m["ttfa_ms"] = round((stamps["first_audio"] - se) * 1000, 1)
            m["first_content_audio_ms"] = round((first_content - se) * 1000, 1)
        await self._emit_bg({"type": "metrics", **m})
