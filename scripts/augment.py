"""Deterministic audio augmentation for the eval scenes. int16 in, int16 out. Every function takes an rng."""
import numpy as np
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

BABBLE_RMS = 3000.0   # fixed reference level so the SNR mixer, not the babble, decides loudness


def _f(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float64)


def _i16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), -32768, 32767).astype(np.int16)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def _fit(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Loop / crop noise to n samples from a random start (noise shorter than speech is common: 8 s DEMAND clips)."""
    start = int(rng.integers(0, len(noise)))
    reps = -(-(n + start) // len(noise))
    return np.tile(noise, reps)[start:start + n]


def mix_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    s = _f(speech)
    ps = _rms(s)
    if ps < 1e-9 or len(noise) == 0:
        return speech.copy()                      # silence stays silence, silent noise adds nothing (no NaN)
    n = _fit(_f(noise), len(s), rng)
    pn = _rms(n)
    if pn < 1e-9:
        return speech.copy()
    return _i16(s + n * (ps / (pn * 10 ** (snr_db / 20))))


def others(ids: list[str], exclude: str, k: int, rng: np.random.Generator) -> list[str]:
    """k distinct ids that are not `exclude` (babble must never contain the target speaker)."""
    pool = [i for i in ids if i != exclude]
    if len(pool) < k:
        raise ValueError(f"need {k} other speakers, have {len(pool)}")
    return [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]


def babble(clips: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """Sum of several talkers (each looped / cropped to n samples), at a fixed reference level."""
    mix = np.zeros(n)
    for c in clips:
        x = _f(c)
        r = _rms(x)
        if r > 1e-9:
            mix += _fit(x, n, rng) / r
    r = _rms(mix)
    return _i16(mix * (BABBLE_RMS / r)) if r > 1e-9 else np.zeros(n, np.int16)


def reverb(x: np.ndarray, rt60: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Convolve with a synthetic exponential-decay RIR (direct path + noise tail reaching -60 dB at rt60). Level kept."""
    n = max(int(rt60 * sr), 1)
    t = np.arange(n) / sr
    h = rng.normal(0, 1, n) * np.exp(-6.9078 * t / rt60)
    h[0] = 1.0 + 0.0 * h[0]
    h = h / np.sqrt(np.sum(h * h))
    y = fftconvolve(_f(x), h)[:len(x)]
    r_in, r_out = _rms(_f(x)), _rms(y)
    return _i16(y * (r_in / r_out)) if r_out > 1e-9 else x.copy()


def gain(x: np.ndarray, db: float) -> np.ndarray:
    return _i16(_f(x) * 10 ** (db / 20))          # clips at int16 limits (the +12 dB "loud" scene)


def telephone(x: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Handset band (300-3400 Hz) + 8 kHz round-trip, for wideband clips played as if over a phone line."""
    sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    y = sosfilt(sos, _f(x))
    if sr != 8000:
        g = np.gcd(sr, 8000)
        y = resample_poly(resample_poly(y, 8000 // g, sr // g), sr // g, 8000 // g)
    y = y[:len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return _i16(y)
