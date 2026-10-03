import numpy as np
import pytest

from scripts import augment as A

SR = 16000


def tone(f: float, n: int = SR, amp: float = 8000) -> np.ndarray:
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.int16)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def band_energy(x: np.ndarray, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    return float(spec[(freqs >= lo) & (freqs < hi)].sum())


def test_mix_snr_hits_target_within_0_1db():
    s = tone(300)
    n = np.random.default_rng(1).normal(0, 3000, SR).astype(np.int16)
    for snr in (20, 10, 5, 0):
        out = A.mix_snr(s, n, snr, np.random.default_rng(2))
        assert out.dtype == np.int16 and len(out) == len(s)
        added = out.astype(np.float64) - s.astype(np.float64)
        assert abs(20 * np.log10(rms(s) / rms(added)) - snr) < 0.1


def test_mix_snr_noise_shorter_than_speech_loops():
    s = tone(300)
    n = np.random.default_rng(1).normal(0, 3000, 100).astype(np.int16)   # 100 samples vs 16000
    out = A.mix_snr(s, n, 10, np.random.default_rng(0))
    assert len(out) == len(s)
    added = out.astype(np.float64) - s.astype(np.float64)
    assert abs(20 * np.log10(rms(s) / rms(added)) - 10) < 0.1


def test_mix_snr_silent_speech_no_nan():
    out = A.mix_snr(np.zeros(SR, np.int16), tone(440), 10, np.random.default_rng(0))
    assert out.dtype == np.int16 and not out.any()          # silence stays silence, never NaN / garbage


def test_mix_snr_silent_noise_returns_speech():
    s = tone(300)
    out = A.mix_snr(s, np.zeros(1000, np.int16), 10, np.random.default_rng(0))
    assert np.array_equal(out, s)


def test_babble_uses_other_speakers_only():
    ids = [f"spk{i}" for i in range(10)]
    for seed in range(20):
        pick = A.others(ids, "spk3", 4, np.random.default_rng(seed))
        assert len(pick) == 4 and len(set(pick)) == 4 and "spk3" not in pick
    with pytest.raises(ValueError):
        A.others(["spk3", "spk4"], "spk3", 4, np.random.default_rng(0))   # not enough other speakers


def test_babble_length_level_and_inputs_untouched():
    clips = [tone(200 + 50 * i, 8000) for i in range(4)]
    keep = [c.copy() for c in clips]
    b = A.babble(clips, 20000, np.random.default_rng(0))
    assert len(b) == 20000 and b.dtype == np.int16 and b.any()
    assert all(np.array_equal(c, k) for c, k in zip(clips, keep))   # not mutated
    assert abs(rms(b) - 3000) < 400                                  # fixed reference level


def test_reverb_keeps_length_energy_bounded_and_is_deterministic():
    x = tone(500)
    y1 = A.reverb(x, 0.6, SR, np.random.default_rng(0))
    y2 = A.reverb(x, 0.6, SR, np.random.default_rng(0))
    assert len(y1) == len(x) and y1.dtype == np.int16
    assert np.array_equal(y1, y2)
    assert 0.5 < rms(y1) / rms(x) < 2.0
    assert not np.array_equal(y1, A.reverb(x, 0.6, SR, np.random.default_rng(1)))   # RIR depends on the rng


def test_reverb_rt60_adds_tail_energy():
    burst = np.zeros(SR, np.int16)
    burst[:400] = tone(800, 400)
    short = A.reverb(burst, 0.2, SR, np.random.default_rng(0))
    long = A.reverb(burst, 0.9, SR, np.random.default_rng(0))
    assert rms(long[SR // 2:]) > rms(short[SR // 2:])


def test_gain_clip_in_int16_range():
    loud = A.gain(tone(300, SR, 20000), 12)
    assert loud.dtype == np.int16 and loud.max() == 32767 and loud.min() == -32768
    quiet = A.gain(tone(300, SR, 20000), -30)
    assert abs(rms(quiet) / rms(tone(300, SR, 20000)) - 10 ** (-30 / 20)) < 0.01


def test_telephone_band_limits():
    noise = np.random.default_rng(0).normal(0, 3000, SR * 2).astype(np.int16)
    out = A.telephone(noise, SR)
    assert len(out) == len(noise) and out.dtype == np.int16
    inband = band_energy(out, 300, 3400)
    assert band_energy(out, 4200, 8000) / inband < 1e-3     # < -30 dB above the 8 kHz-sampling Nyquist
    assert band_energy(out, 0, 100) / inband < 1e-3         # < -30 dB below the handset band


def test_telephone_odd_length_keeps_length():
    assert len(A.telephone(tone(300, 12345), SR)) == 12345
