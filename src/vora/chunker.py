"""Turn a token stream into TTS-sized chunks. First chunk is split early on a clause to cut time-to-first-audio."""
import re

_END = "。！？!?\n"
_CLAUSE = "，,；;、"
_ABBR = {"e.g", "i.e", "vs", "mr", "mrs", "dr", "etc", "no"}
_CJK = re.compile(r"[一-鿿]")


def _is_cjk(s: str) -> bool:
    return len(_CJK.findall(s)) * 2 >= max(1, len(s.strip()))


class SentenceChunker:
    def __init__(self, first_words: int = 4, second_words: int = 0) -> None:
        self.second_words = second_words   # 0 = off; else the 2nd chunk is cut after this many words, hiding a short 1st chunk
        self.second_done = False
        self.first_words = first_words   # first audio chunk = this many Latin words (synth time ~ 0.2x audio length)
        self.buf = ""
        self.first_done = False

    def _emit(self, n: int, out: list[str]) -> None:
        chunk, self.buf = self.buf[:n].strip(), self.buf[n:]
        if chunk:
            if self.first_done:
                self.second_done = True
            out.append(chunk)
            self.first_done = True

    def _period_ends_sentence(self, i: int) -> bool | None:
        """True/False, or None when undecidable until the next token arrives."""
        if i + 1 >= len(self.buf):
            return None  # need one more char (decimal / abbreviation); flush() covers the final sentence
        if self.buf[i - 1].isdigit() and self.buf[i + 1].isdigit():
            return False
        if self.buf[i + 1] not in " \n":
            return False
        word = re.split(r"\s", self.buf[:i])[-1].lower()
        return word not in _ABBR

    def _clause_long_enough(self, i: int) -> bool:
        head = self.buf[: i + 1]
        return len(head.strip()) >= (8 if _is_cjk(head) else 12)

    def _scan_cut(self) -> int | None:
        """First sentence end, or (before the first chunk) the first clause break that is long enough."""
        for i, ch in enumerate(self.buf):
            if ch in _END:
                return i + 1
            if ch == ".":
                ends = self._period_ends_sentence(i)
                if ends is None:
                    return None
                if ends:
                    return i + 1
            elif ch in _CLAUSE and not self.first_done and self._clause_long_enough(i):
                return i + 1
        return None

    def _word_cut(self, n: int) -> int | None:
        w = self.buf.split(" ")
        if len(w) >= n + 1 and len(" ".join(w[:n])) >= int(3.5 * n):   # "You can" (7) must split, "Yes it" (6) must not
            return len(" ".join(w[:n])) + 1
        return None

    def _limit_cut(self) -> int | None:
        cjk = _is_cjk(self.buf)
        limit = 60 if cjk else 120
        if len(self.buf) < limit:
            return None
        if cjk:
            return limit
        sp = self.buf.rfind(" ", 0, limit)
        return sp + 1 if sp > 0 else limit

    def _next_cut(self) -> int | None:
        cut = self._scan_cut()
        # latency: the first audio chunk is synthesised at ~0.2x real time, so keep it to ~first_words words
        if cut is None and not self.first_done and not _is_cjk(self.buf):
            cut = self._word_cut(self.first_words)
        if cut is None and self.first_done and not self.second_done and self.second_words and not _is_cjk(self.buf):
            cut = self._word_cut(self.second_words)
        return cut if cut is not None else self._limit_cut()

    def push(self, token: str) -> list[str]:
        self.buf += token
        out: list[str] = []
        while (cut := self._next_cut()) is not None:
            self._emit(cut, out)
        return out

    def flush(self) -> list[str]:
        out: list[str] = []
        self._emit(len(self.buf), out)
        return out
