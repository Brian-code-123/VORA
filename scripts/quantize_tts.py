"""Mixed-precision INT8 dynamic quantization of Piper VITS ONNX voices.
The flows and the text encoder hold ~85% of the weight bytes but little of the compute; the HiFi-GAN decoder (`/dec/`) holds
~10% of the bytes and nearly all the compute. Quantizing everything (ConvInteger) made the first chunk ~3x slower on arm64
(LJSpeech medium: p95 179-199 ms vs 64 ms fp32). Quantizing everything except `/dec/` keeps 22 MB (limit 30) at p95 ~80 ms.
The output keeps the `.int8.onnx` name the loader looks for.
Usage: quantize_tts.py models/tts_en_ljspeech/en_US-ljspeech-medium.onnx"""
import sys
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

KEEP_FP32 = ("/dec/",)


def quantize(src: Path, keep_fp32: tuple[str, ...] = KEEP_FP32) -> Path:
    dst = src.with_suffix(".int8.onnx")
    skip = [n.name for n in onnx.load(str(src)).graph.node if n.name.startswith(keep_fp32)]
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8, nodes_to_exclude=skip)
    print(f"{src.name}: {src.stat().st_size/1e6:.1f} MB -> {dst.stat().st_size/1e6:.1f} MB ({len(skip)} decoder nodes kept fp32)")
    return dst


if __name__ == "__main__":
    for p in sys.argv[1:]:
        quantize(Path(p))
