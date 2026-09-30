"""Download pinned models into models/. Free HF repos only. Writes MANIFEST.json (sha256 + license)."""
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
REV = {'csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17': 'd42f2d9f7ca24806fb667456a18a9f1b60f70d16', 'csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23': '204ad334e2e683fd295359930cc16fc0432a23ac', 'csukuangfj/vits-piper-en_US-lessac-low': '7f5edb341cadf35daab356f857ed5ae8545d22bf', 'csukuangfj/vits-piper-zh_CN-huayan-x_low': 'bf8284cf534fd7328467afd3c8b3eb992eedc02a'}  # pinned commits (checked 2026-10-01)
LLM_REV = "9217f5db79a29953eb74d5343926648285ec7e67"
ROOT = Path(__file__).resolve().parent.parent / "models"
SPECS = {
    "asr_en": dict(repo="csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17", license="Apache-2.0",
                   allow=["*.int8.onnx", "tokens.txt", "test_wavs/*"], ignore=["*epoch-99-avg-1.onnx"]),
    "asr_zh": dict(repo="csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23", license="Apache-2.0",
                   allow=["*.int8.onnx", "tokens.txt", "test_wavs/*"], ignore=[]),
    "tts_en": dict(repo="csukuangfj/vits-piper-en_US-lessac-low", license="Piper MIT; voice dataset: Blizzard 2013 Lessac licence (custom, not OSI/CC)",
                   allow=["*.onnx", "tokens.txt", "espeak-ng-data/*", "MODEL_CARD"], ignore=[]),
    "tts_zh": dict(repo="csukuangfj/vits-piper-zh_CN-huayan-x_low", license="Piper MIT; voice dataset licence UNKNOWN per Piper model card",
                   allow=["*.onnx", "tokens.txt", "lexicon.txt", "espeak-ng-data/*", "*.fst", "*.txt"], ignore=[]),
}
LLM = dict(repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF", file="qwen2.5-0.5b-instruct-q4_k_m.gguf", license="Apache-2.0")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    manifest = {}
    for name, s in SPECS.items():
        d = ROOT / name
        snapshot_download(s["repo"], revision=REV[s["repo"]], local_dir=d, allow_patterns=s["allow"], ignore_patterns=s["ignore"] or None)
        if name == "tts_en" and not list(d.glob("*.int8.onnx")):   # 30 MB gate: quantize the 63 MB fp32 voice (needs onnx)
            from quantize_tts import quantize
            quantize(next(p for p in d.glob("*.onnx") if not p.name.endswith(".int8.onnx")))
        manifest[name] = {"repo": s["repo"], "license": s["license"],
                          "files": {str(f.relative_to(d)): sha(f) for f in sorted(d.rglob("*")) if f.is_file() and f.suffix in {".onnx", ".gguf"}}}
    d = ROOT / "llm"
    p = hf_hub_download(LLM["repo"], LLM["file"], revision=LLM_REV, local_dir=d)
    manifest["llm"] = {"repo": LLM["repo"], "license": LLM["license"], "files": {LLM["file"]: sha(Path(p))}}
    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("ok")


if __name__ == "__main__":
    main()
