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

from vora.agc import Agc
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
MIN_ECHO_CHUNK = 10         # letters/digits: a spoken chunk shorter than this is not evidence that the mic heard our speaker
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
        self.context_ready = False      # retrieval done; whoever finishes second (retrieval or promotion) sends the context event
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


class _TurnRun:
    """State and worker threads of one answer. `llm_worker` and `tts_worker` run in executors and reach the event loop only
    through run_coroutine_threadsafe. `gate` is set for a shadow turn: its output is held back until it is promoted."""

    def __init__(self, pipe: "Pipeline", text: str, hits, trace: LatencyTrace, ev: threading.Event, gate: ShadowGate | None) -> None:
        self.pipe, self.text, self.hits, self.trace, self.ev, self.gate = pipe, text, hits, trace, ev, gate
        self.sent_q: queue.Queue = queue.Queue()
        self.answer: list[str] = []      # every token, in order: the screen text must not depend on the droppable token messages
        self.stamps: dict[str, float] = gate.stamps if gate is not None else {}
        self.admitted = False
        self._released = threading.Event()
        self._rendered = 0

    def stamp(self, is_filler: bool) -> None:
        now = self.pipe.clock()
        self.stamps.setdefault("first_audio", now)
        if not is_filler:
            self.stamps.setdefault("first_content", now)

    def record_spoken(self, sentence: str) -> None:
        if self.gate is not None:
            with self.gate.lock:
                if not self.gate.open.is_set():
                    self.gate.spoken.append((self.pipe.clock(), sentence))
                    return
        self.pipe._spoken.append((self.pipe.clock(), sentence))

    def _say(self, sentence: str, is_filler: bool) -> None:
        self.record_spoken(sentence)
        self.sent_q.put((sentence, is_filler))

    def say_filler(self) -> None:
        self._say(FILLER[_lang(self.text)], True)

    def release_once(self) -> None:   # exactly one release per acquire, whichever path ends the turn first
        if not self._released.is_set():
            self._released.set()
            self.pipe.ex.admission.release()

    def _hold_token(self, tok: str) -> bool:
        """True while a shadow turn is muted: the token waits in the gate instead of going to the client."""
        if self.gate is None:
            return False
        with self.gate.lock:
            held = not self.gate.open.is_set()
            if held:
                self.gate.tokens.append(tok)
        return held

    def _stream_answer(self, chunker: SentenceChunker) -> None:
        for tok in self.pipe.llm.stream(self.text, self.hits, self.ev):
            if self.ev.is_set():
                return
            self.answer.append(tok)
            if "first_token" not in self.trace.marks:
                self.trace.mark("first_token")
            if not self._hold_token(tok):
                asyncio.run_coroutine_threadsafe(
                    self.pipe._emit_bg({"type": "token", "text": tok, "full": "".join(self.answer)}, droppable=True), self.pipe.loop)
            for c in chunker.push(tok):
                self._say(c, False)
        for c in chunker.flush():
            self._say(c, False)

    def llm_worker(self) -> None:
        if self.ev.is_set():
            self.release_once()
            return
        chunker = SentenceChunker(first_words=self.pipe.s.first_chunk_words, second_words=self.pipe.s.second_chunk_words)
        try:
            self._stream_answer(chunker)
        except Exception:  # noqa: BLE001 - a failing model must not kill the session: log it and say so
            log.exception("LLM turn failed")
            self.sent_q.put((BUSY[_lang(self.text)], False))
        finally:
            self.release_once()
            self.sent_q.put(None)

    def busy_worker(self) -> None:
        self.sent_q.put((BUSY[_lang(self.text)], False))
        self.sent_q.put(None)

    def emit_audio(self, chunk: bytes, is_filler: bool) -> None:
        self.stamp(is_filler)
        asyncio.run_coroutine_threadsafe(self.pipe._put_audio(chunk, self.ev), self.pipe.loop).result()

    def flush_buffered(self) -> None:
        for pc, pf in self.gate.take_audio():
            self.emit_audio(pc, pf)

    def _flush_if_promoted(self) -> None:
        if self.gate is not None and self.gate.open.is_set():
            self.flush_buffered()

    def _wait_for_promotion(self) -> bool:
        """Shadow turn: pre-render only the first sentences, then wait for the final text. False if cancelled meanwhile."""
        gate = self.gate
        self._rendered += 1
        while not gate.open.is_set() and self._rendered > SHADOW_PRERENDER_SENTENCES and not self.ev.is_set():
            gate.open.wait(0.01)
        if self.ev.is_set():
            return False
        if gate.open.is_set():
            self.flush_buffered()
        return True

    def _speak(self, sentence: str, is_filler: bool) -> bool:
        """Synthesise one sentence. False when the turn was cancelled meanwhile."""
        gate = self.gate
        if gate is not None and not self._wait_for_promotion():
            return False
        for chunk in self.pipe.tts.synth(sentence, self.ev):
            if self.ev.is_set():
                return False
            if gate is not None:
                with gate.lock:
                    if not gate.open.is_set():
                        gate.audio.append((chunk, is_filler))
                        continue
                self.flush_buffered()                          # anything pre-rendered goes out first
            self.emit_audio(chunk, is_filler)
        return True

    def tts_worker(self) -> None:
        while not self.ev.is_set():
            try:
                item = self.sent_q.get(timeout=0.01 if self.gate is not None else 0.05)
            except queue.Empty:
                self._flush_if_promoted()               # promoted while we were idle: release the pre-rendered audio now
                continue
            if item is None:
                self._flush_if_promoted()
                return
            if not self._speak(*item):
                return


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
        self._agc = Agc() if settings.agc else None   # quiet microphones: boost before the voice detector and ASR see the audio
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
        recent = [r for r in (_alnum(s) for ts, s in self._spoken if now - ts < ECHO_WINDOW_S) if r]
        if not recent:
            return False
        if t in "".join(recent):                                   # what the mic heard is a stretch of what we said
            return True
        # A spoken chunk inside the heard text proves nothing when it is short: the first TTS chunk is one word ("The"), which
        # made any follow-up containing that word an "echo" and silently dropped it.
        return any((len(r) >= MIN_ECHO_CHUNK and r in t) or t in r or difflib.SequenceMatcher(None, t, r).ratio() >= 0.75 for r in recent)

    async def on_audio(self, pcm: bytes) -> None:
        if self.stalled:
            raise ClientStalled()
        now = self.clock()
        if self._agc is not None:
            pcm = self._agc.process(pcm)
        if self._voiced(pcm):
            self._last_voice = now   # plan: speech_end = last voiced input frame
        events = await self.loop.run_in_executor(self.ex.asr, self.asr.feed, pcm)
        if self._shadow is not None and self.active_sessions() > 1:
            await self._cancel_shadow()                    # another user arrived: do not burn their CPU on a guess
        for e in events:
            if e.kind == "partial":
                await self._on_partial(e)
            else:
                await self._on_final(e.text, getattr(e, "held_ms", 0.0))

    async def _on_partial(self, e) -> None:
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
            await self._maybe_barge_in(e.text)

    async def _maybe_barge_in(self, text: str) -> None:
        while self._change_times and self.clock() - self._change_times[0] > BARGE_IN_WINDOW_S:
            self._change_times.popleft()
        if len(self._change_times) >= BARGE_IN_CHANGES and not self._is_echo(text):
            await self._cancel_turn()  # user talks over the reply

    def _turn_active(self) -> bool:
        return self._turn is not None and not self._turn.done()

    async def wait_idle(self) -> None:
        if self._turn is not None:
            try:
                await self._turn
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():   # we were cancelled, not the turn
                    raise

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
            await self._cancel_shadow()                # a guess built on our own speaker's echo
            await self._emit({"type": "echo_ignored", "text": text})
            return
        if self._shadow is not None and _same_text(self._shadow.text, text) and not self._turn_active() \
                and not self._shadow.task.done():       # a refused / failed guess has nothing to promote
            await self._promote_shadow(text, speech_end)
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
        cancelled_ourselves = False
        try:
            await self._turn
        except asyncio.CancelledError:
            cancelled_ourselves = bool(asyncio.current_task().cancelling())
        self._drain_audio()
        if announce:  # never block on a full queue: the client must learn about the cancel
            while self.out.full():
                self.out.get_nowait()
            self.out.put_nowait(("json", {"type": "cancel"}))
        if cancelled_ourselves:   # cleanup done; do not swallow the caller's own cancellation
            raise asyncio.CancelledError

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
        if not self._shadow_wanted(text) or self._is_echo(text):
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

    async def _promote_shadow(self, text: str, speech_end: float) -> None:
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
            ctx_ready = g.context_ready
        self._spoken.extend(spoken)
        if ctx_ready:
            await self._emit_bg({"type": "context", "ids": g.context_ids})
        acc = ""
        for t in tokens:
            acc += t
            await self._emit_bg({"type": "token", "text": t, "full": acc}, droppable=True)   # `full`: a dropped token leaves no hole
        # buffered audio is released by the TTS thread (the only writer) or, if it already finished, by _run_turn

    async def _announce_context(self, hits, gate: ShadowGate | None) -> None:
        if gate is None:
            await self._emit_bg({"type": "context", "ids": [h.chunk_id for h in hits]})
            return
        with gate.lock:
            gate.context_ids, gate.context_ready = [h.chunk_id for h in hits], True
            promoted = gate.open.is_set()
        if promoted:            # promoted while we were still retrieving: _promote_shadow left the event to us
            await self._emit_bg({"type": "context", "ids": gate.context_ids})

    async def _run_turn(self, text: str, trace: LatencyTrace, ev: threading.Event, prefetched, held_ms: float = 0.0,
                        gate: ShadowGate | None = None) -> None:
        hits = await self._hits_for(text, prefetched)
        await self._announce_context(hits, gate)
        run = _TurnRun(self, text, hits, trace, ev, gate)
        if self.s.filler:
            run.say_filler()
        run.admitted = self.ex.admission.try_acquire()
        if gate is not None and not run.admitted:
            return                                     # a guess never queues behind (or talks over) a real turn
        try:
            await asyncio.gather(
                self.loop.run_in_executor(self.ex.llm, run.llm_worker) if run.admitted else self.loop.run_in_executor(None, run.busy_worker),
                self.loop.run_in_executor(self.ex.tts, run.tts_worker),
            )
        finally:
            if run.admitted:
                run.release_once()   # cancelled before the worker started, or it died: free the slot
        if gate is not None:
            await self._drain_gate(run)
        if ev.is_set():
            return
        await self._report_turn(run, held_ms)

    async def _drain_gate(self, run: "_TurnRun") -> None:
        gate = run.gate
        while not gate.open.is_set():              # finished early (short answer): wait to be promoted or cancelled
            await asyncio.sleep(0.01)
        if not run.ev.is_set():
            for pc, pf in gate.take_audio():       # the TTS thread is gone: this coroutine is the only writer
                run.stamp(pf)
                await self._put_audio(pc, run.ev)

    async def _report_turn(self, run: "_TurnRun", held_ms: float) -> None:
        trace, stamps, gate = run.trace, run.stamps, run.gate
        first_content = stamps.get("first_content", stamps.get("first_audio"))
        if first_content is not None:
            trace.marks["first_audio"] = first_content  # content audio is what the brief's 1.5 s means
        await self._emit_bg({"type": "answer", "text": "".join(run.answer)})   # complete text; the live tokens above may have been dropped
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
