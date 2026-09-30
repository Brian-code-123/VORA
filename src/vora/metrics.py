import time
from typing import Callable


class LatencyTrace:
    """Marks in order: speech_end, final, first_token, first_audio. report() in ms."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self._clock = clock
        self.marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        self.marks[name] = self._clock()

    def _d(self, a: str, b: str) -> float | None:
        if a in self.marks and b in self.marks:
            return round((self.marks[b] - self.marks[a]) * 1000, 1)
        return None

    def report(self) -> dict[str, float]:
        pairs = {
            "asr_final": ("speech_end", "final"),
            "rag_first_token": ("final", "first_token"),
            "tts_first_chunk": ("first_token", "first_audio"),
            "total": ("speech_end", "first_audio"),
        }
        out = {k: self._d(*v) for k, v in pairs.items()}
        return {k: v for k, v in out.items() if v is not None}
