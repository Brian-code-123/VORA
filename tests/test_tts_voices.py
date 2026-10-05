import json
from pathlib import Path

import numpy as np
import pytest

from vora.config import ROOT, Settings
from vora.tts import SR, Tts, en_voice_dir

S = Settings()
LJ = S.models_dir / "tts_en_ljspeech"


def test_voice_dir_mapping():
    assert en_voice_dir(Settings(tts_en_voice="ljspeech")).name == "tts_en_ljspeech"
    assert en_voice_dir(Settings(tts_en_voice="lessac")).name == "tts_en"          # the old location keeps the lessac voice


def test_default_english_voice_is_the_permissive_one():
    assert Settings().tts_en_voice == "ljspeech"


@pytest.mark.skipif(not LJ.exists(), reason="LJSpeech voice not fetched")
def test_ljspeech_int8_le_30mb():
    int8 = next(LJ.glob("*.int8.onnx"))
    assert int8.stat().st_size <= 30 * 1024 * 1024


@pytest.mark.skipif(not LJ.exists(), reason="LJSpeech voice not fetched")
def test_en_voice_output_16k_mono_and_duration_preserved():
    tts = Tts(Settings(tts_en_voice="ljspeech"))
    real, grabbed = tts.voices["en"], []

    class Spy:                    # VITS durations are stochastic: compare against the very audio synth() resampled
        sample_rate = real.sample_rate

        def generate(self, *a, **kw):
            out = real.generate(*a, **kw)
            grabbed.append(out)
            return out
    tts.voices["en"] = Spy()
    pcm = np.frombuffer(b"".join(tts.synth("Hello there, this is a test.")), dtype=np.int16)
    native = grabbed[0]
    assert native.sample_rate == 22050            # medium voice: resampled to 16 kHz for the client
    assert abs(len(pcm) / SR - len(native.samples) / native.sample_rate) < 0.005
    assert pcm.dtype == np.int16 and np.abs(pcm).max() > 1000


@pytest.mark.skipif(not LJ.exists(), reason="LJSpeech voice not fetched")
def test_voice_switch_setting_loads_the_other_model():
    if not (S.models_dir / "tts_en").exists():
        pytest.skip("lessac voice not present")
    a = Tts(Settings(tts_en_voice="ljspeech")).voices["en"].sample_rate
    b = Tts(Settings(tts_en_voice="lessac")).voices["en"].sample_rate
    assert (a, b) == (22050, 16000)


def test_manifest_en_voice_licence_permissive():
    m = json.loads((S.models_dir / "MANIFEST.json").read_text()) if (S.models_dir / "MANIFEST.json").exists() else {}
    if "tts_en_ljspeech" not in m:
        pytest.skip("MANIFEST.json not regenerated yet (python scripts/fetch_models.py)")
    lic = m["tts_en_ljspeech"]["license"].lower()
    assert "public domain" in lic and "custom" not in lic and "unknown" not in lic


def test_licenses_md_lists_the_voices_and_the_zh_exception():
    text = (ROOT / "docs" / "licenses.md").read_text(encoding="utf-8").lower()
    assert "ljspeech" in text and "public domain" in text
    assert "huayan" in text and "exception" in text           # zh voice: licence unknown, kept by ruling, documented
    assert "espeak-ng" in text and "gpl" in text              # the phonemizer data is GPL-3.0: said out loud


def test_licenses_md_mentions_every_manifest_model():
    mp = S.models_dir / "MANIFEST.json"
    if not mp.exists():
        pytest.skip("no manifest")
    text = (ROOT / "docs" / "licenses.md").read_text(encoding="utf-8").lower()
    for name, spec in json.loads(mp.read_text()).items():
        assert spec["repo"].split("/")[-1].lower() in text, f"docs/licenses.md does not mention {spec['repo']}"


def _tiny_vits_like(path: Path) -> None:
    """Two MatMuls with weights big enough to quantize: one under /enc_p/ (text encoder), one under /dec/ (audio decoder)."""
    import onnx
    from onnx import TensorProto, helper
    rng = np.random.default_rng(0)
    inits = [helper.make_tensor(n, TensorProto.FLOAT, [64, 64], rng.normal(size=64 * 64).astype(np.float32).tolist()) for n in ("w_enc", "w_dec")]
    nodes = [helper.make_node("MatMul", ["x", "w_enc"], ["h"], name="/enc_p/encoder/mm"),
             helper.make_node("MatMul", ["h", "w_dec"], ["y"], name="/dec/resblocks.0/mm")]
    g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64])], inits)
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]), str(path))


def test_quantize_keeps_the_audio_decoder_in_fp32(tmp_path):
    """Quantizing the HiFi-GAN decoder (small weights, nearly all the compute) made the first chunk 3x slower on arm64;
    the flows + text encoder hold ~85% of the bytes. So only those are quantized."""
    import onnx
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from quantize_tts import quantize
    src = tmp_path / "v.onnx"
    _tiny_vits_like(src)
    out = quantize(src)
    ops = {n.name: n.op_type for n in onnx.load(str(out)).graph.node}
    assert ops["/enc_p/encoder/mm_quant"] in ("MatMulInteger", "DynamicQuantizeLinear") or any(o == "MatMulInteger" for n, o in ops.items() if n.startswith("/enc_p/"))
    assert ops["/dec/resblocks.0/mm"] == "MatMul"            # untouched


@pytest.mark.skipif(not LJ.exists(), reason="LJSpeech voice not fetched")
def test_ljspeech_mixed_precision_le_30mb_and_decoder_fp32():
    import onnx
    int8 = next(LJ.glob("*.int8.onnx"))
    assert int8.stat().st_size <= 30 * 1024 * 1024
    m = onnx.load(str(int8))
    assert not any(n.op_type in ("ConvInteger", "MatMulInteger", "QLinearConv") for n in m.graph.node if n.name.startswith("/dec/"))
    assert any(n.op_type in ("ConvInteger", "MatMulInteger") for n in m.graph.node)      # and it really is quantized elsewhere


@pytest.mark.perf
@pytest.mark.skipif(not LJ.exists(), reason="LJSpeech voice not fetched")
def test_en_first_chunk_p95_le_200ms_ljspeech():
    import time
    tts = Tts(Settings(tts_en_voice="ljspeech"))
    list(tts.synth("Hello."))
    firsts = ["The", "Blue", "Hold", "You", "Please", "Firmware", "Return", "Sure", "Hello", "Yes", "Two", "Mandarin", "Reset", "About", "Support", "Privacy", "Wake", "Power", "Speaker", "Wi-Fi"]
    t = []
    for w in firsts * 2:
        t0 = time.perf_counter()
        next(iter(tts.synth(w + " ")))
        t.append((time.perf_counter() - t0) * 1000)
    t.sort()
    assert t[int(0.95 * len(t)) - 1] <= 200, t


def test_resampler_keeps_a_tone_and_the_length():
    from vora.tts import resample
    t = np.arange(22050) / 22050
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.5
    y = resample(x, 22050, 16000)
    assert len(y) == 16000
    ref = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000)
    err = y[1000:-1000] - ref[1000:-1000]
    assert 10 * np.log10(np.mean(ref[1000:-1000] ** 2) / np.mean(err ** 2)) > 40      # > 40 dB SNR
    assert len(resample(np.zeros(0, np.float32), 22050, 16000)) == 0


def test_tts_runtime_does_not_import_scipy():
    import subprocess, sys
    code = "import sys; import vora.tts, vora.pipeline, vora.server; print('scipy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**__import__('os').environ, "PYTHONPATH": "src"}).stdout.strip()
    assert out == "False"        # scipy.signal alone cost 72 MB USS on Linux arm64
