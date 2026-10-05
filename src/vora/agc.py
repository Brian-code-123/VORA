"""Automatic gain control in front of the voice detector and ASR. Quiet input (−30 dB) raised LibriSpeech-clean WER from
5.1% to 42.9% and hid speech from the RMS voice detector. Only boosts (never attenuates), only follows frames that stand
out from the noise floor, caps the gain at +30 dB and keeps the amplified noise floor below ~-35 dBFS."""
import numpy as np

TARGET_RMS = 3000.0          # ~ -21 dBFS, what the eval sets' speech sits at
MAX_GAIN = 31.6              # +30 dB
MIN_SPEECH_RMS = 30.0        # below this a frame is never speech (digital hiss)
MAX_NOISE_AFTER_GAIN = 600.0


class Agc:
    def __init__(self):
        self.floor: float | None = None
        self.level: float | None = None
        self.gain = 1.0

    def process(self, pcm: bytes) -> bytes:
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if not x.size:
            return pcm
        r = float(np.sqrt(np.mean(x * x)))
        self.floor = r if self.floor is None else min(self.floor * 1.01 + 0.5, r)    # same floor rule as the voice detector
        if r > max(3.0 * self.floor, MIN_SPEECH_RMS):
            self.level = r if self.level is None or r > self.level else 0.85 * self.level + 0.15 * r   # fast attack, ~1 s release
        want = 1.0 if self.level is None else min(max(TARGET_RMS / self.level, 1.0), MAX_GAIN)
        want = min(want, max(1.0, MAX_NOISE_AFTER_GAIN / max(self.floor, 1e-3)))
        self.gain += (want - self.gain) * 0.5
        if self.gain < 1.001:
            self.gain = 1.0
            return pcm                                       # normal input passes through bit-exact
        peak = float(np.abs(x).max())
        g = min(self.gain, 0.9 * 32767 / peak) if peak > 0 else self.gain   # never boost a frame into clipping
        if g <= 1.0:
            return pcm
        return np.clip(np.rint(x * g), -32767, 32767).astype(np.int16).tobytes()
