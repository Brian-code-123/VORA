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

    def push(self, token: str) -> list[str]:
        self.buf += token
        out: list[str] = []
        while True:
            cut = None
            for i, ch in enumerate(self.buf):
                if ch in _END:
                    cut = i + 1
                    break
                if ch == ".":
                    r = self._period_ends_sentence(i)
                    if r is None:
                        break
                    if r:
                        cut = i + 1
                        break
                elif ch in _CLAUSE and not self.first_done:
                    head = self.buf[: i + 1]
                    if len(head.strip()) >= (8 if _is_cjk(head) else 12):
                        cut = i + 1
                        break
            if cut is None and not self.first_done and not _is_cjk(self.buf):
                # latency: the first audio chunk is synthesised at ~0.2x real time, so keep it to ~first_words words
                n = self.first_words
                w = self.buf.split(" ")
                if len(w) >= n + 1 and len(" ".join(w[:n])) >= int(3.5 * n):   # "You can" (7) must split, "Yes it" (6) must not
                    cut = len(" ".join(w[:n])) + 1
            if cut is None and self.first_done and not self.second_done and self.second_words and not _is_cjk(self.buf):
                w = self.buf.split(" ")
                if len(w) >= self.second_words + 1 and len(" ".join(w[: self.second_words])) >= int(3.5 * self.second_words):
                    cut = len(" ".join(w[: self.second_words])) + 1
            if cut is None:
                limit = 60 if _is_cjk(self.buf) else 120
                if len(self.buf) >= limit:
                    cut = limit
                    if not _is_cjk(self.buf):
                        sp = self.buf.rfind(" ", 0, limit)
                        cut = sp + 1 if sp > 0 else limit
                else:
                    return out
            self._emit(cut, out)

    def flush(self) -> list[str]:
        out: list[str] = []
        self._emit(len(self.buf), out)
        return out
