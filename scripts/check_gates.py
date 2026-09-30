"""Size gates from the brief. Exit 0 only if every present model passes."""
import sys
from pathlib import Path

MB = 1024 * 1024
LIMITS = {"asr_en": 50 * MB, "asr_zh": 50 * MB, "tts_en": 30 * MB, "tts_zh": 30 * MB}
REPORT_ONLY = {"llm", "embed"}  # LLM bounded by params (<=1B), not MB; RSS reported separately


def evaluate(sizes: dict[str, int]) -> dict[str, dict]:
    out = {}
    for k, v in sizes.items():
        if k in REPORT_ONLY:
            out[k] = {"size_mb": v / MB, "limit_mb": None, "ok": True}
        else:
            out[k] = {"size_mb": v / MB, "limit_mb": LIMITS[k] / MB, "ok": v <= LIMITS[k]}
    return out


def measure(models_dir: Path) -> dict[str, int]:
    """Count only the weights the runtime loads (int8 onnx / gguf), not espeak data."""
    sizes = {}
    for d in sorted(models_dir.iterdir()) if models_dir.exists() else []:
        if not d.is_dir():
            continue
        files = [f for f in d.rglob("*") if f.is_file()]
        if d.name.startswith("asr"):
            files = [f for f in files if f.suffix == ".onnx" and ".int8" in f.name or f.name.startswith(("decoder", "joiner")) and ".int8" in f.name]
        elif d.name.startswith("tts"):
            q = [f for f in files if f.name.endswith(".int8.onnx")]
            files = q or [f for f in files if f.suffix == ".onnx"]
        elif d.name == "llm":
            files = [f for f in files if f.suffix == ".gguf"]
        elif d.name == "embed":
            files = [f for f in files if f.suffix == ".onnx"]
        sizes[d.name] = sum(f.stat().st_size for f in files)
    return sizes


if __name__ == "__main__":
    res = evaluate(measure(Path(__file__).resolve().parent.parent / "models"))
    bad = False
    for k, v in res.items():
        lim = "n/a" if v["limit_mb"] is None else f"{v['limit_mb']:.0f}"
        print(f"{k:8s} {v['size_mb']:8.1f} MB  limit {lim:>4s}  {'OK' if v['ok'] else 'FAIL'}")
        bad |= not v["ok"]
    sys.exit(1 if bad else 0)
