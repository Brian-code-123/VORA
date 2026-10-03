import re
from pathlib import Path
import threading
from typing import Iterator

from llama_cpp import Llama

from vora.config import Settings
from vora.guard import focus_on_asked_product, product_codes, split_sentences, best_sentence, expects_number, extractive_answer, has_number, is_digitish, is_refusal, is_yes_no_question, numbers_mismatch, polarity_conflict, is_question_echo
from vora.rag.store import Hit

SYSTEM = (
    "Answer from the context only, in at most two short sentences, in the question's language "
    "(Simplified Chinese for Chinese). If the context lacks it, say you are not sure."
)
_CJK = re.compile(r"[一-鿿]")
UNSURE = {"zh": "我不确定，请换个问法。", "en": "I'm not sure about that. Could you rephrase?"}
BUSY = {"zh": "系统正忙，请稍后再试。", "en": "The system is busy, please try again."}
_DECIDE_TOKENS = 6   # tokens buffered before the guard / extractive-fallback check (was 10)


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


def reading_text(mode: str, question: str, hit: Hit) -> str:
    """The text the model reads (and the guards check against) for one hit. "chunk": the whole chunk minus the sentences about
    a product the question did not ask for. "focus": only the retriever's best sentence, unless that sentence is about the
    other product."""
    asked = product_codes(question)
    if mode == "focus" and hit.focus and (not asked or not product_codes(hit.focus) or product_codes(hit.focus) & asked):
        return hit.focus
    return focus_on_asked_product(question, hit.text)


def chatml_prompt(msgs: list[dict], think_off: bool) -> str:
    """ChatML text for models whose chat template we bypass. think_off: Qwen3 starts with a reasoning block; an empty one
    makes it answer directly (and keeps <think> tokens out of the stream the guard reads)."""
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + "<|im_start|>assistant\n"
    return out + "<think>\n\n</think>\n\n" if think_off else out


class Llm:
    """Qwen2.5-0.5B GGUF (default) or Qwen3-0.6B (Settings.llm_dir). One instance per process; llama.cpp is not thread-safe, so a lock serialises users."""

    @staticmethod
    def model_path(settings: Settings) -> Path:
        d = settings.models_dir / settings.llm_dir
        found = sorted(d.glob("*.gguf")) if d.exists() else []
        if not found:
            raise FileNotFoundError(f"no .gguf in {d} (Settings.llm_dir={settings.llm_dir!r}, fetch it first)")
        return found[0]

    def __init__(self, settings: Settings):
        self.s = settings
        self.llm = Llama(
            model_path=str(self.model_path(settings)),
            n_ctx=1024, n_threads=settings.llm_threads, n_threads_batch=settings.llm_threads,  # default n_threads_batch = all logical cores: E-core stragglers made prefill 10x slower (22 vs 374 tok/s)
            n_batch=512, n_gpu_layers=0, verbose=False,
        )
        # No LlamaRAMCache: every entry copies ~90 MB of logits (RSS 2.5 GB, seconds of memcpy). llama.cpp already reuses the
        # KV prefix of the previous prompt, which is the shared system prompt when users alternate.
        self._think_off = self.llm.metadata.get("general.architecture") == "qwen3"
        self._lock = threading.Lock()
        self._cancel = threading.Event()

    @staticmethod
    def _replacement(question: str, hits: list[Hit], head: str, ref: set[str]) -> str | None:
        """None = keep the model's answer. Else the text sentence to speak instead: the answer ignored the context,
        refused although we have hits, said yes to a negated fact (any hit), or invented a number."""
        chunks = [h.text for h in hits]
        if any(polarity_conflict(question, c, head) for c in chunks) or numbers_mismatch(" ".join(chunks), head) \
                or not (_terms(head) & ref) or is_refusal(head):
            return extractive_answer(question, chunks)
        return None

    def cancel(self) -> None:
        self._cancel.set()

    def _ntok(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode(), add_bos=False))

    def _context(self, question: str, hits: list[Hit]) -> str:
        """Top hit whole if it fits the budget, else its sentences that best match the question (original order);
        more hits only if they still fit. Keeps prefill (and first-token time) short without cutting the answer."""
        budget, parts, used = self.s.max_ctx_tokens, [], 0
        for i, h in enumerate(hits):
            n = self._ntok(h.text)
            if used + n <= budget:
                parts.append(h.text)
                used += n
            elif i == 0:
                sents = split_sentences(h.text)
                best = best_sentence(h.text, question)
                keep = {sents.index(best)} if best in sents else {0}
                used += self._ntok(sents[next(iter(keep))])
                for j in sorted(range(len(sents)), key=lambda j: abs(j - next(iter(keep)))):   # grow outwards from the best one
                    if j in keep:
                        continue
                    m = self._ntok(sents[j])
                    if used + m > budget:
                        break
                    keep.add(j)
                    used += m
                parts.append(" ".join(sents[j] for j in sorted(keep)))
            else:
                break
        return "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(parts))

    def _messages(self, question: str, hits: list[Hit]) -> list[dict]:
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Context:\n{self._context(question, hits)}\n\nQuestion: {question}"}]

    def _deltas(self, msgs: list[dict]) -> Iterator[str | None]:
        if self._think_off:
            for ev in self.llm.create_completion(prompt=chatml_prompt(msgs, True), stream=True, max_tokens=80, temperature=self.s.llm_temperature,
                                                 stop=["<|im_end|>"]):
                yield ev["choices"][0]["text"]
        else:
            for ev in self.llm.create_chat_completion(messages=msgs, stream=True, max_tokens=80, temperature=self.s.llm_temperature):
                yield ev["choices"][0]["delta"].get("content")

    def prompt_tokens(self, question: str, hits: list[Hit]) -> int:
        return sum(self._ntok(m["content"]) for m in self._messages(question, hits)) + 20   # chat-template overhead

    def stream(self, question: str, hits: list[Hit], cancel: threading.Event | None = None) -> Iterator[str]:
        """`cancel` is a per-turn event (shared instance flag would let one user's barge-in kill another's reply)."""
        lang = _lang(question)
        if not hits:
            yield UNSURE[lang]
            return
        hits = [Hit(h.chunk_id, reading_text(self.s.context_mode, question, h), h.score) for h in hits]
        if self.s.yes_no_extractive and is_yes_no_question(question):
            # a 0.5B model ignores negation in yes/no questions (measured: "yes, understands Cantonese"); quote the text
            yield extractive_answer(question, [h.text for h in hits])
            return
        if not self._lock.acquire(timeout=3.0):
            yield BUSY[lang]
            return
        if cancel is None:
            cancel = self._cancel
            cancel.clear()
        gen = None
        try:
            msgs = self._messages(question, hits)
            gen = self._deltas(msgs)
            buf, decided, ref = [], False, _terms(hits[0].text) | _terms(question)
            pending: list[str] = []          # digit-ish tokens held back until the whole number is known
            chunks_text = " ".join(h.text for h in hits)

            said: list[str] = []             # everything yielded so far (a number is judged in context: "X" + "200" is a model code)

            def number_ok() -> bool:
                return not numbers_mismatch(chunks_text, "".join(said) + "".join(pending))

            def decide() -> str | None:
                """Head = buffered tokens minus a trailing, possibly incomplete number (that moves to `pending`)."""
                while buf and is_digitish(buf[-1]):
                    pending.insert(0, buf.pop())
                return self._replacement(question, hits, "".join(buf), ref)

            for tok in gen:
                if cancel.is_set():
                    return
                if not tok:
                    continue
                if not decided:
                    buf.append(tok)
                    if len(buf) < _DECIDE_TOKENS:
                        continue
                    decided = True
                    fix = decide()
                    if fix is not None:
                        yield fix
                        return
                    said.append("".join(buf))
                    yield "".join(buf)
                    continue
                if is_digitish(tok):
                    pending.append(tok)
                    continue
                if pending:
                    if not number_ok():                      # invented figure: replace the rest with the real sentence
                        yield extractive_answer(question, [h.text for h in hits])
                        return
                    said.append("".join(pending))
                    yield "".join(pending)
                    pending.clear()
                said.append(tok)
                yield tok
            if not decided and buf:  # short answer that ended before the decision point
                fix = decide()
                text = "".join(buf) + "".join(pending)
                yield text if fix is None and (not pending or number_ok()) else extractive_answer(question, [h.text for h in hits])
            elif pending:
                yield "".join(pending) if number_ok() else extractive_answer(question, [h.text for h in hits])
            if decided and not cancel.is_set() and not expects_number(question) and is_question_echo("".join(said) + "".join(pending), question, chunks_text):
                yield " " + extractive_answer(question, [h.text for h in hits])   # the whole answer only rephrased the question: add the text
            if decided and not cancel.is_set() and expects_number(question) and not has_number("".join(said) + "".join(pending)) \
                    and has_number(chunks_text):
                yield " " + extractive_answer(question, [h.text for h in hits])   # asked "how far/long/much": the figure is in the text
        finally:
            if gen is not None:
                gen.close()
            self._lock.release()
