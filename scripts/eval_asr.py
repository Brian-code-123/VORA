"""ASR accuracy through the real streaming wrapper. WER (en) / CER (zh), clean and 10 dB white-noise.
Data streamed from HF: LibriSpeech test-clean, FLEURS cmn_hans_cn + en_us test. Usage: eval_asr.py [--n 50]"""
import argparse
import io
import json
import re
import time

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from jiwer import cer, wer

from vora.asr import AsrSession, load_recognizers
from vora.config import ROOT, Settings

SR = 16000
SETS = {
    "en_librispeech_clean": ("openslr/librispeech_asr", "clean", "en", "text"),
    "en_fleurs": ("google/fleurs", "en_us", "en", "transcription"),
    "zh_fleurs": ("google/fleurs", "cmn_hans_cn", "zh", "transcription"),
}


def norm(t: str, lang: str) -> str:
    t = t.lower()
    if lang == "zh":
        return re.sub(r"[\s\W_]+", "", t)
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", t).split())


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p = np.mean(x.astype(np.float64) ** 2) + 1e-9
    n = rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), x.shape)
    return np.clip(x + n, -32768, 32767).astype(np.int16)


def transcribe(recs, lang: str, pcm: np.ndarray) -> tuple[str, float]:
    sess = AsrSession(recs, lang)
    x = np.concatenate([pcm, np.zeros(int(1.5 * SR), dtype=np.int16)])
    t0, finals = time.perf_counter(), []
    for i in range(0, len(x), 1600):
        finals += [e.text for e in sess.feed(x[i:i + 1600].tobytes()) if e.kind == "final"]
    return ("" if lang == "zh" else " ").join(finals), time.perf_counter() - t0


def load(name: str, n: int):
    """Streams once from HF, then caches to data/eval_cache/<name>.npz (HF streaming is flaky)."""
    cache = ROOT / "data" / "eval_cache" / f"{name}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(zip(z["refs"].tolist(), z["pcm"]))[:n]
    for attempt in range(4):
        try:
            out = _stream(name, n)
            break
        except Exception as e:  # noqa: BLE001 - network flakiness
            print("retry", attempt, type(e).__name__)
            time.sleep(3)
    else:
        raise RuntimeError(f"could not stream {name}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, refs=np.array([r for r, _ in out], dtype=object), pcm=np.array([p for _, p in out], dtype=object))
    return out


def _stream(name: str, n: int):
    repo, cfg, lang, col = SETS[name]
    ds = load_dataset(repo, cfg, split="test", streaming=True).cast_column("audio", Audio(decode=False))
    out = []
    for ex in ds:
        x, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")  # FLEURS is float32 wav; dtype=int16 reads it as silence
        if x.ndim > 1:
            x = x[:, 0]
        assert sr == SR, sr
        pcm = (x / (np.abs(x).max() + 1e-9) * 0.5 * 32767).astype(np.int16)  # peak-normalise to -6 dBFS (FLEURS peaks ~0.07)
        if len(pcm) / SR <= 20:  # keep clips short so one utterance = one reference
            out.append((ex[col], pcm))
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    recs = load_recognizers(Settings())
    rng = np.random.default_rng(0)
    res = {}
    for name, (_, _, lang, _) in SETS.items():
        data = load(name, a.n)
        for cond in ("clean", "noisy_10dB"):
            refs, hyps, proc, dur = [], [], 0.0, 0.0
            for ref, pcm in data:
                x = pcm if cond == "clean" else add_noise(pcm, 10, rng)
                h, dt = transcribe(recs, lang, x)
                refs.append(norm(ref, lang)); hyps.append(norm(h, lang))
                proc += dt; dur += len(x) / SR + 1.5
            err = (cer if lang == "zh" else wer)(refs, hyps)
            res[f"{name}/{cond}"] = {"metric": "CER" if lang == "zh" else "WER", "value": round(err, 4), "n": len(refs),
                                     "rtf": round(proc / dur, 3), "example": {"ref": refs[0], "hyp": hyps[0]}}
            print(name, cond, res[f"{name}/{cond}"]["metric"], round(err, 4), "RTF", round(proc / dur, 3))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "asr.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
