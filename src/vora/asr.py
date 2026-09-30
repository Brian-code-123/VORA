from dataclasses import dataclass
from typing import Literal

import numpy as np
import sherpa_onnx

from vora.config import Settings

SR = 16000
MAX_UTTERANCE_S = 30


@dataclass(frozen=True)
class AsrEvent:
    kind: Literal["partial", "final"]
    text: str
    stable: bool = False


def load_recognizers(settings: Settings) -> dict[str, "sherpa_onnx.OnlineRecognizer"]:
    out = {}
    for lang in ("en", "zh"):
        d = settings.models_dir / f"asr_{lang}"
        out[lang] = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"),
            encoder=str(next(d.glob("encoder*.int8.onnx"))),
            decoder=str(next(d.glob("decoder*.int8.onnx"))),
            joiner=str(next(d.glob("joiner*.int8.onnx"))),
            num_threads=settings.asr_threads,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=0.6,  # sherpa default 2.4 s would burn the 1.5 s budget
            rule2_min_trailing_silence=0.4,  # default 1.2 s
            rule3_min_utterance_length=MAX_UTTERANCE_S,
        )
    return out


def _norm(t: str) -> str:
    return " ".join(t.lower().split())


class AsrSession:
    """One user's ASR state. feed() is synchronous: call it from an executor, never the event loop."""

    def __init__(self, recognizers: dict, lang: Literal["zh", "en"] = "en"):
        self.rec = recognizers[lang]
        self.lang = lang
        self._new_stream()

    def _new_stream(self) -> None:
        self.stream = self.rec.create_stream()
        self._last = ""
        self._same = 0
        self._samples = 0
        # 0.5 s pre-roll: the streaming zipformer drops the first words of audio that starts abruptly
        self.stream.accept_waveform(SR, np.zeros(SR // 2, dtype=np.float32))

    def reset(self) -> None:
        self._new_stream()

    def feed(self, pcm16: bytes) -> list[AsrEvent]:
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if x.size == 0:
            return []
        self._samples += x.size
        self.stream.accept_waveform(SR, x)
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        r = self.rec.get_result(self.stream)
        text = _norm(getattr(r, "text", r))
        events: list[AsrEvent] = []
        if text and text != self._last:
            self._same = 0
            events.append(AsrEvent("partial", text, False))
        elif text:
            self._same += 1
            if self._same == 2:
                events.append(AsrEvent("partial", text, True))
        self._last = text
        if self.rec.is_endpoint(self.stream) or self._samples > MAX_UTTERANCE_S * SR:
            if text:
                events.append(AsrEvent("final", text, True))
            self.rec.reset(self.stream)
            self._last, self._same, self._samples = "", 0, 0
        return events
