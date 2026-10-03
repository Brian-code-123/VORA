"""Reference voice-activity detection, dev tool only (never in the live path).
The pipeline defines speech_end as the last voiced 100 ms frame by an adaptive-floor RMS rule. On real noisy clips that rule
could keep firing on background noise and make latency look better than it is, so every clip is cross-checked against
silero VAD (MIT). Usage: python scripts/refvad.py --fetch"""
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SILERO_PATH = ROOT / "models" / "aux" / "silero_vad.onnx"
SILERO_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
SILERO_SHA256 = "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"
SR = 16000
FRAME = 1600   # the pipeline sees 100 ms frames


def pipeline_speech_end(pcm: np.ndarray) -> float:
    """Seconds at which the pipeline would stamp speech_end for this audio fed in real time (end of the last voiced frame).
    Runs Pipeline._voiced itself so the rule can never drift from the live code."""
    from vora.pipeline import Pipeline
    state = SimpleNamespace(_floor=None)
    last = 0.0
    for i in range(0, len(pcm), FRAME):
        if Pipeline._voiced(state, pcm[i:i + FRAME].tobytes()):
            last = (i + len(pcm[i:i + FRAME])) / SR
    return last


def speech_end(pcm: np.ndarray) -> float:
    """Seconds at which silero says the last speech segment ends (0.0 when it finds none)."""
    import sherpa_onnx
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(SILERO_PATH)
    cfg.silero_vad.min_silence_duration = 0.25
    cfg.silero_vad.min_speech_duration = 0.1
    cfg.sample_rate = SR
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
    x = pcm.astype(np.float32) / 32768.0
    win = cfg.silero_vad.window_size
    for i in range(0, len(x) - win + 1, win):
        vad.accept_waveform(x[i:i + win])
    vad.flush()
    end = 0.0
    while not vad.empty():
        seg = vad.front
        end = max(end, (seg.start + len(seg.samples)) / SR)
        vad.pop()
    return end


def fetch() -> None:
    import urllib.request
    SILERO_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SILERO_PATH.with_suffix(".tmp")
    urllib.request.urlretrieve(SILERO_URL, tmp)
    got = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if got != SILERO_SHA256:
        tmp.unlink()
        raise SystemExit(f"silero_vad.onnx sha256 mismatch: {got}")
    tmp.replace(SILERO_PATH)
    print("ok", SILERO_PATH)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    if "--fetch" in sys.argv:
        fetch()
