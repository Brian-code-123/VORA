"""Download pinned models into models/. Free HF repos only. Writes MANIFEST.json (sha256 + license)."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent / "models"
SPECS = {
    "asr_en": dict(repo="csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17", license="Apache-2.0",
                   allow=["*.int8.onnx", "tokens.txt", "test_wavs/*"], ignore=["*epoch-99-avg-1.onnx"]),
    "asr_zh": dict(repo="csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23", license="Apache-2.0",
                   allow=["*.int8.onnx", "tokens.txt", "test_wavs/*"], ignore=[]),
    "tts_en": dict(repo="csukuangfj/vits-piper-en_US-lessac-low", license="MIT (Piper) / voice dataset public domain",
                   allow=["*.onnx", "tokens.txt", "espeak-ng-data/*", "MODEL_CARD"], ignore=[]),
    "tts_zh": dict(repo="csukuangfj/vits-piper-zh_CN-huayan-x_low", license="MIT (Piper)",
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
        snapshot_download(s["repo"], local_dir=d, allow_patterns=s["allow"], ignore_patterns=s["ignore"] or None)
        manifest[name] = {"repo": s["repo"], "license": s["license"],
                          "files": {str(f.relative_to(d)): sha(f) for f in sorted(d.rglob("*")) if f.is_file() and f.suffix in {".onnx", ".gguf"}}}
    d = ROOT / "llm"
    p = hf_hub_download(LLM["repo"], LLM["file"], local_dir=d)
    manifest["llm"] = {"repo": LLM["repo"], "license": LLM["license"], "files": {LLM["file"]: sha(Path(p))}}
    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("ok")


if __name__ == "__main__":
    main()
