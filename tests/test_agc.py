import numpy as np

from vora.agc import Agc

SR = 16000


def tone(seconds, amp, f=300.0):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.int16)


def run(agc, x):
    return np.concatenate([np.frombuffer(agc.process(x[i:i + 1600].tobytes()), dtype=np.int16) for i in range(0, len(x), 1600)])


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def test_quiet_speech_is_brought_up_within_a_second():
    lead = np.random.default_rng(1).normal(0, 2, SR // 2).astype(np.int16)    # half a second of room tone first, as in real input
    y = run(Agc(), np.concatenate([lead, tone(2.0, 100)]))                      # ~ -50 dBFS speech-level tone
    assert rms(y[SR + SR // 2:]) > 1500                                         # within 1 s it is at a normal level (target ~3000)


def test_normal_and_loud_speech_untouched():
    x = tone(1.0, 6000)
    assert np.array_equal(run(Agc(), x), x)
    loud = tone(1.0, 30000)
    assert np.array_equal(run(Agc(), loud), loud)       # never attenuates (the clip is already there)


def test_silence_and_noise_floor_not_amplified():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 3, SR * 2).astype(np.int16)   # digital hiss
    assert rms(run(Agc(), noise)) < 10


def test_gain_capped_and_no_wraparound():
    y = run(Agc(), np.concatenate([tone(1.0, 20), tone(0.2, 20000)]))
    assert y.dtype == np.int16 and np.abs(y.astype(np.int32)).max() <= 32767
    assert rms(y[:SR]) <= 20 * 31.7 + 1                  # at most +30 dB


def test_length_and_dtype_preserved():
    x = tone(0.35, 500)
    assert len(run(Agc(), x)) == len(x)


def test_boost_never_pushes_a_frame_into_clipping():
    """Loud clipped speech has quiet syllables between loud ones; boosting those must not create new clipping
    (measured: loud +12 dB input WER 7.1% -> 8.5% before the peak limit)."""
    agc = Agc()
    run(agc, tone(0.5, 25000))                      # loud speech sets the level
    for _ in range(30):                              # a long quiet stretch lets the gain rise
        run(agc, tone(0.1, 900))
    y = run(agc, tone(0.3, 12000))                   # a medium syllable right after: may be boosted, never clipped
    assert np.abs(y.astype(np.int32)).max() <= int(0.9 * 32767) + 1
