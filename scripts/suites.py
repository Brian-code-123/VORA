"""Public eval suites -> data/suites/<name>/{manifest.jsonl, wav/*.wav, filter_report.json}. Third-party speech only.
Every row carries source + licence (eval/suites/pins.json pins dataset revisions).
Usage: python scripts/suites.py build <suite|all> ...      python scripts/suites.py list"""
import hashlib
import io
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
SUITES_DIR = ROOT / "data" / "suites"
PINS = ROOT / "eval" / "suites" / "pins.json"
SR = 16000
MIN_S, MAX_S = 0.8, 20.0
FIELDS = ("id", "wav", "text", "lang", "scenario", "accent", "intent", "source", "licence", "split", "orig_sr", "aug")

MINDS14 = {"minds14_en_us": "en-US", "minds14_en_gb": "en-GB", "minds14_en_au": "en-AU", "minds14_zh": "zh-CN"}
READ = {   # suite -> (eval_asr set name, dataset pin key, scenario)
    "librispeech_clean": ("en_librispeech_clean", "librispeech", "read_clean"),
    "librispeech_other": ("en_librispeech_other", "librispeech", "read_hard"),
    "fleurs_en": ("en_fleurs", "fleurs", "read_hard"),
    "fleurs_zh": ("zh_fleurs", "fleurs", "read_hard"),
    "aishell": ("zh_aishell", "aishell", "read_clean"),
}
NOISE = "ambient_noise"
READ_N = 200


def pins() -> dict:
    return json.loads(PINS.read_text(encoding="utf-8"))["datasets"]


def split_of(id: str) -> str:
    return "dev" if int(hashlib.md5(id.encode()).hexdigest(), 16) % 2 == 0 else "test"


def make_row(*, id: str, wav: str, text: str, lang: str, scenario: str, accent: str, intent: str | None,
             source: str, licence: str, orig_sr: int, aug: str = "clean") -> dict:
    if not source or not licence:
        raise ValueError(f"row {id}: source and licence are required")
    return {"id": id, "wav": wav, "text": text, "lang": lang, "scenario": scenario, "accent": accent, "intent": intent,
            "source": source, "licence": licence, "split": split_of(id), "orig_sr": orig_sr, "aug": aug}


def decode_audio(b: bytes) -> tuple[np.ndarray, int] | None:
    try:
        x, sr = sf.read(io.BytesIO(b), dtype="float32")
    except Exception:  # noqa: BLE001 - any decoder failure = skip the clip, counted in the report
        return None
    return (x[:, 0] if x.ndim > 1 else x), sr


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz mono int16, peak-normalised to -6 dBFS (same convention as the other eval sets)."""
    y = x.astype(np.float64)
    if x.dtype == np.int16:
        y /= 32768.0
    if sr != SR:
        g = gcd(sr, SR)
        y = resample_poly(y, SR // g, sr // g)
    peak = float(np.abs(y).max()) if y.size else 0.0
    if peak < 1e-9:
        return np.zeros(len(y), np.int16)
    return np.clip(np.rint(y / peak * 0.5 * 32767), -32768, 32767).astype(np.int16)


def keep(text: str, dur_s: float) -> str | None:
    """None = keep, otherwise the reason this clip is dropped."""
    if not text.strip():
        return "empty_text"
    if dur_s < MIN_S:
        return "too_short"
    if dur_s > MAX_S:
        return "too_long"
    return None


def _write(out_dir: Path, id: str, pcm: np.ndarray) -> str:
    rel = f"wav/{id}.wav"
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / rel, pcm, SR, subtype="PCM_16")
    return rel


def rows_from_minds14(table, cfg: str, intent_names: list[str], out_dir: Path) -> tuple[list[dict], dict]:
    p = pins()["minds14"]
    lang = "zh" if cfg.startswith("zh") else "en"
    report = {"seen": 0, "kept": 0, "corrupt": 0, "empty_text": 0, "too_short": 0, "too_long": 0}
    rows = []
    for r in table.to_pylist():
        report["seen"] += 1
        dec = decode_audio(r["audio"]["bytes"])
        if dec is None:
            report["corrupt"] += 1
            continue
        x, sr = dec
        why = keep(r["transcription"], len(x) / sr)
        if why:
            report[why] += 1
            continue
        id = f"minds14_{cfg}_{Path(r['path']).stem}"
        rel = _write(out_dir, id, to_16k(x, sr))
        rows.append(make_row(id=id, wav=rel, text=r["transcription"].strip(), lang=lang, scenario="telephone", accent=cfg,
                             intent=intent_names[r["intent_class"]], source=p["repo"], licence=p["licence"], orig_sr=sr))
        report["kept"] += 1
    return rows, report


def _fetch_minds14(cfg: str):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = pins()["minds14"]
    path = hf_hub_download(p["repo"], f"{cfg}/train-00000-of-00001.parquet", repo_type="dataset", revision=p["revision"])
    table = pq.read_table(path)
    names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["intent_class"]["names"]
    return table, names


def _read_rows(name: str, out_dir: Path) -> tuple[list[dict], dict]:
    from scripts import eval_asr
    set_name, pin_key, scenario = READ[name]
    p = pins()[pin_key]
    lang = eval_asr.SETS[set_name][2]
    report = {"seen": 0, "kept": 0, "empty_text": 0, "too_short": 0, "too_long": 0}
    rows = []
    for i, (text, pcm) in enumerate(eval_asr.load(set_name, READ_N, tag=f"_n{READ_N}")):
        report["seen"] += 1
        why = keep(text, len(pcm) / SR)
        if why:
            report[why] += 1
            continue
        id = f"{name}_{i:04d}"
        rows.append(make_row(id=id, wav=_write(out_dir, id, pcm), text=text.strip(), lang=lang, scenario=scenario, accent="",
                             intent=None, source=p["repo"], licence=p["licence"], orig_sr=SR))
        report["kept"] += 1
    return rows, report


def _noise_rows(out_dir: Path) -> tuple[list[dict], dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = pins()["demand"]
    path = hf_hub_download(p["repo"], "data/noise-00000-of-00001.parquet", repo_type="dataset", revision=p["revision"])
    rows, bad = [], 0
    for i, r in enumerate(pq.read_table(path).to_pylist()):
        dec = decode_audio(r["audio"]["bytes"])
        if dec is None:
            bad += 1
            continue
        x, sr = dec
        id = f"ambient_{i:02d}"
        rows.append(make_row(id=id, wav=_write(out_dir, id, to_16k(x, sr)), text="", lang="", scenario="ambient", accent="", intent=None,
                             source=p["repo"], licence=p["licence"], orig_sr=sr))
    return rows, {"seen": len(rows) + bad, "kept": len(rows), "corrupt": bad}


def build(name: str) -> Path:
    out = SUITES_DIR / name
    if (out / "manifest.jsonl").exists():
        return out
    if name in MINDS14:
        table, names = _fetch_minds14(MINDS14[name])
        rows, report = rows_from_minds14(table, MINDS14[name], names, out)
    elif name in READ:
        rows, report = _read_rows(name, out)
    elif name == NOISE:
        rows, report = _noise_rows(out)
    else:
        raise KeyError(f"unknown suite: {name}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "filter_report.json").write_text(json.dumps(report, indent=1))
    (out / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return out


def load_manifest(name: str) -> list[dict]:
    p = SUITES_DIR / name / "manifest.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"suite {name} not built: python scripts/suites.py build {name}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_wav(name: str, row: dict) -> np.ndarray:
    x, sr = sf.read(SUITES_DIR / name / row["wav"], dtype="int16")
    assert sr == SR, sr
    return x


def latency_subset(rows: list[dict], split: str = "test", per_en: int = 1, per_zh: int = 3) -> list[str]:
    """Fixed, hash-ordered ids for the real-time latency runs: per (accent, intent) `per_en` clips (en) or `per_zh` (zh)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["split"] == split and r["intent"]:
            groups.setdefault((r["accent"], r["intent"]), []).append(r)
    out = []
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda r: hashlib.md5(r["id"].encode()).hexdigest())
        out += [r["id"] for r in g[:per_zh if g[0]["lang"] == "zh" else per_en]]
    return out


def write_latency_ids() -> dict:
    """Freeze the latency subsets into pins.json so a code change cannot silently swap the clips."""
    ids = {}
    for lang, names in (("en", ("minds14_en_us", "minds14_en_gb", "minds14_en_au")), ("zh", ("minds14_zh",))):
        rows = [r for n in names for r in load_manifest(n)]
        ids[lang] = {s: latency_subset(rows, s) for s in ("dev", "test")}
    doc = json.loads(PINS.read_text(encoding="utf-8"))
    doc["latency_ids"] = ids
    PINS.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ids


ALL = [*MINDS14, *READ, NOISE]


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("build", "list"):
        sys.exit(__doc__)
    if argv[0] == "list":
        print("\n".join(ALL))
        return
    names = ALL if argv[1:] == ["all"] else argv[1:]
    for n in names:
        p = build(n)
        print(n, "->", p, json.loads((p / "filter_report.json").read_text()))
    if names == ALL:
        print({k: {s: len(v) for s, v in d.items()} for k, d in write_latency_ids().items()})


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main(sys.argv[1:])
