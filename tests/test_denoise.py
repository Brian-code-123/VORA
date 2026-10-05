"""The GTCRN enhancer is a dev-only experiment (scripts/denoise_ab.py): it made WER worse in every scene, so it is not in the
live path. These tests keep the wrapper honest in case the experiment is re-run."""
import numpy as np
import pytest

from scripts.denoise import MODEL, Denoiser

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="models/aux/gtcrn_simple.onnx not fetched")


def test_length_preserved_after_flush_and_dtype():
    d = Denoiser()
    x = (np.random.default_rng(0).normal(0, 2000, 16000)).astype(np.int16)
    out = b"".join(d.process(x[i:i + 1600].tobytes()) for i in range(0, len(x), 1600)) + d.flush()
    y = np.frombuffer(out, dtype=np.int16)
    assert abs(len(y) - len(x)) <= 256


def test_silence_stays_silent():
    d = Denoiser()
    out = np.frombuffer(d.process(np.zeros(3200, np.int16).tobytes()) + d.flush(), dtype=np.int16)
    assert np.abs(out).max() < 100
