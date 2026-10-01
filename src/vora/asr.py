import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import sherpa_onnx

from vora.config import Settings
from vora.endpoint import looks_incomplete, strip_disfluency

SR = 16000
MAX_UTTERANCE_S = 30


@dataclass(frozen=True)
class AsrEvent:
    kind: Literal["partial", "final"]
    text: str
    stable: bool = False
    held_ms: float = 0.0   # final only: how long the endpoint was held because the text looked unfinished


def _endpoint_kwargs(settings: Settings) -> dict:
    return dict(
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=0.6,  # sherpa default 2.4 s would burn the 1.5 s budget
        rule2_min_trailing_silence=settings.endpoint_rule2_s,  # default 1.2 s
        rule3_min_utterance_length=MAX_UTTERANCE_S,
    )


def _transducer(d, settings: Settings):
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(d / "tokens.txt"),
        encoder=str(next(d.glob("encoder*.int8.onnx"))),
        decoder=str(next(d.glob("decoder*.int8.onnx"))),
        joiner=str(next(d.glob("joiner*.int8.onnx"))),
        num_threads=settings.asr_threads,
        decoding_method="modified_beam_search",  # WER 26.6% -> 16.6% vs greedy (LibriSpeech, 50 clips)
        **_endpoint_kwargs(settings),
    )


def _zipformer2_ctc(d, settings: Settings):
    return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
        tokens=str(d / "tokens.txt"),
        model=str(d / "model.int8.onnx"),
        num_threads=settings.asr_threads,
        decoding_method="greedy_search",   # CTC: greedy only
        **_endpoint_kwargs(settings),
    )


def load_recognizers(settings: Settings) -> dict[str, "sherpa_onnx.OnlineRecognizer"]:
    zh = _zipformer2_ctc(settings.models_dir / "asr_zh_ctc", settings) if settings.asr_zh_model == "ctc_small" \
        else _transducer(settings.models_dir / "asr_zh", settings)
    return {"en": _transducer(settings.models_dir / "asr_en", settings), "zh": zh}


def _norm(t: str) -> str:
    return " ".join(t.lower().split())


class AsrSession:
    """One user's ASR state. feed() is synchronous: call it from an executor, never the event loop.

    The recognizer stream is NOT reset at an endpoint: a reset throws away the encoder context and cost ~6 CER points
    (AISHELL-1 10.9% vs 4.7%; second utterances 31-37% WER). Instead each final is the text added since the previous
    final. The stream is only reset after a final once it holds more than RESET_AFTER_S of audio."""

    RESET_AFTER_S = 20

    def __init__(self, recognizers: dict, lang: Literal["zh", "en"] = "en", hold_ms: int = 500, hold_total_cap_ms: int = 1200):
        self.rec = recognizers[lang]
        self.lang = lang
        self.hold_ms, self.hold_total_cap_ms = hold_ms, hold_total_cap_ms
        self._new_stream()

    def _new_stream(self) -> None:
        self.stream = self.rec.create_stream()
        self._reset_state()
        # 0.8 s pre-roll: the streaming zipformer drops the first words of audio that starts abruptly
        # (sweep on 50 LibriSpeech clips, beam search: lead 0 s WER 16.6%, 0.3 s 14.0%, 0.8 s 8.2%)
        self.stream.accept_waveform(SR, np.zeros(SR * 4 // 5, dtype=np.float32))

    def _reset_state(self) -> None:
        self._last = ""          # last partial shown (normalised)
        self._same = 0
        self._emitted = ""       # raw recognizer text already sent as finals
        self._stream_samples = 0
        self._since_final = 0
        self._hold_since = None   # stream-sample mark where the current hold began
        self._hold_text = ""
        self._held = 0            # samples held for the current utterance (capped)

    def reset(self) -> None:
        self._new_stream()

    def _raw(self) -> str:
        r = self.rec.get_result(self.stream)
        return getattr(r, "text", r)

    def feed(self, pcm16: bytes) -> list[AsrEvent]:
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if x.size == 0:
            return []
        self._stream_samples += x.size
        self._since_final += x.size
        self.stream.accept_waveform(SR, x)
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        raw = self._raw()
        new = raw[len(self._emitted):] if raw.startswith(self._emitted) else raw   # CTC/transducer text is append-only
        text = _norm(new)
        events: list[AsrEvent] = []
        if text and text != self._last:
            self._same = 0
            events.append(AsrEvent("partial", text, False))
        elif text:
            self._same += 1
            if self._same == 2:
                events.append(AsrEvent("partial", text, True))
        self._last = text
        endpoint = self.rec.is_endpoint(self.stream)
        forced = self._since_final > MAX_UTTERANCE_S * SR
        clean = strip_disfluency(text, self.lang)
        if text and (endpoint or forced):
            if not forced and self.hold_ms > 0 and looks_incomplete(clean, self.lang) and self._held < self.hold_total_cap_ms * SR // 1000:
                # user stopped mid-sentence: keep decoding the SAME stream and wait up to hold_ms of audio for more words
                if self._hold_since is None or text != self._hold_text:
                    self._hold_since, self._hold_text = self._stream_samples, text
                self._held += x.size
                if self._stream_samples - self._hold_since < self.hold_ms * SR // 1000:
                    return events
            held_ms = round(self._held * 1000 / SR, 1)
            self._hold_since, self._hold_text, self._held = None, "", 0
            if len(re.sub(r"\W+", "", clean)) >= 2:   # a 1-letter "s" or a lone "uh" must not start a turn
                events.append(AsrEvent("final", clean, True, held_ms=held_ms))
            self._emitted, self._last, self._same, self._since_final = raw, "", 0, 0
            if self._stream_samples > self.RESET_AFTER_S * SR:
                self.rec.reset(self.stream)
                self._reset_state()
        elif endpoint and self._stream_samples > self.RESET_AFTER_S * SR and not text:
            self.rec.reset(self.stream)    # long silence-only stretch: bound the stream
            self._reset_state()
        return events
