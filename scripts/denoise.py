"""Optional speech enhancement (GTCRN, sherpa-onnx OnlineSpeechDenoiser) in front of ASR. Off by default: see
docs/rulings.md for the measured effect per scene."""
from pathlib import Path

import numpy as np
import sherpa_onnx

MODEL = Path(__file__).resolve().parents[1] / "models" / "aux" / "gtcrn_simple.onnx"


class Denoiser:
    """PCM16 bytes in, PCM16 bytes out (streaming: output lags by ~20 ms; flush() returns the tail)."""

    def __init__(self, model: Path = MODEL, threads: int = 1):
        cfg = sherpa_onnx.OnlineSpeechDenoiserConfig(model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
            gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=str(model)), num_threads=threads))
        self._d = sherpa_onnx.OnlineSpeechDenoiser(cfg)

    @staticmethod
    def _out(audio) -> bytes:
        x = np.asarray(audio.samples, dtype=np.float32)
        return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()

    def process(self, pcm: bytes) -> bytes:
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
        return self._out(self._d.run(x, 16000)) if x.size else b""

    def flush(self) -> bytes:
        return self._out(self._d.flush())

    def reset(self) -> None:
        self._d.reset()
