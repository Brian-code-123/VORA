"""INT8 dynamic quantization of Piper VITS ONNX. Usage: quantize_tts.py models/tts_en/en_US-lessac-low.onnx"""
import sys
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize(src: Path) -> Path:
    dst = src.with_suffix(".int8.onnx")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
    print(f"{src.name}: {src.stat().st_size/1e6:.1f} MB -> {dst.stat().st_size/1e6:.1f} MB")
    return dst


if __name__ == "__main__":
    for p in sys.argv[1:]:
        quantize(Path(p))
