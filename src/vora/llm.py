import re
import threading
from typing import Iterator

from llama_cpp import Llama, LlamaRAMCache

from vora.config import Settings
from vora.rag.store import Hit

SYSTEM = (
    "You are the voice assistant of VORA Box. Answer the question using ONLY the context. "
    "Reply in at most two short sentences in the same language as the question "
    "(Simplified Chinese for Chinese questions). Start directly with the answer. "
    "If the context does not contain the answer, say you are not sure."
)
_CJK = re.compile(r"[一-鿿]")
UNSURE = {"zh": "我不确定，请换个问法。", "en": "I'm not sure about that. Could you rephrase?"}
BUSY = {"zh": "系统正忙，请稍后再试。", "en": "The system is busy, please try again."}
_DECIDE_TOKENS = 10  # tokens buffered before the extractive-fallback check


def _lang(text: str) -> str:
    return "zh" if _CJK.search(text) else "en"


def _terms(text: str) -> set[str]:
    t = text.lower()
    words = {w for w in re.findall(r"[a-z0-9]{3,}", t)}
    cjk = "".join(_CJK.findall(t))
    return words | {cjk[i:i + 2] for i in range(len(cjk) - 1)}


def _first_sentence(text: str) -> str:
    m = re.match(r"(.+?[。！？.!?])(\s|$)", text.strip())
    return (m.group(1) if m else text).strip()


class Llm:
    """Qwen2.5-0.5B GGUF. One instance per process; llama.cpp is not thread-safe, so a lock serialises users."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.llm = Llama(
            model_path=str(next((settings.models_dir / "llm").glob("*.gguf"))),
            n_ctx=1024, n_threads=settings.llm_threads, n_batch=128, n_gpu_layers=0, verbose=False,
        )
        self.llm.set_cache(LlamaRAMCache(capacity_bytes=64 << 20))  # 2 alternating users would thrash the single-prompt KV prefix
        self._lock = threading.Lock()
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def _context(self, hits: list[Hit]) -> str:
        parts, used = [], 0
        for h in hits:
            n = len(self.llm.tokenize(h.text.encode()))
            if parts and used + n > self.s.max_ctx_tokens:
                break
            parts.append(h.text)
            used += n
        return "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(parts))

    def stream(self, question: str, hits: list[Hit], cancel: threading.Event | None = None) -> Iterator[str]:
        """`cancel` is a per-turn event (shared instance flag would let one user's barge-in kill another's reply)."""
        lang = _lang(question)
        if not hits:
            yield UNSURE[lang]
            return
        if not self._lock.acquire(timeout=3.0):
            yield BUSY[lang]
            return
        if cancel is None:
            cancel = self._cancel
            cancel.clear()
        gen = None
        try:
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Context:\n{self._context(hits)}\n\nQuestion: {question}"}]
            gen = self.llm.create_chat_completion(messages=msgs, stream=True, max_tokens=80, temperature=0.2)
            buf, decided, ref = [], False, _terms(hits[0].text) | _terms(question)
            for ev in gen:
                if cancel.is_set():
                    return
                tok = ev["choices"][0]["delta"].get("content")
                if not tok:
                    continue
                if decided:
                    yield tok
                    continue
                buf.append(tok)
                if len(buf) >= _DECIDE_TOKENS:
                    decided = True
                    if not (_terms("".join(buf)) & ref):  # extractive fallback: answer ignores the context
                        yield _first_sentence(hits[0].text)
                        return
                    yield "".join(buf)
            if not decided and buf:  # short answer that ended before the decision point
                text = "".join(buf)
                yield text if (_terms(text) & ref) else _first_sentence(hits[0].text)
        finally:
            if gen is not None:
                gen.close()
            self._lock.release()
