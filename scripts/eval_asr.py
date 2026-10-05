"""ASR accuracy through the real streaming wrapper. WER (en) / CER (zh), clean and 10 dB white-noise.
Data streamed from HF: LibriSpeech test-clean, FLEURS cmn_hans_cn + en_us test. Usage: eval_asr.py [--n 50]"""
import argparse
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from jiwer import cer, wer

from vora.asr import AsrSession, load_recognizers
from vora.config import ROOT, Settings

SR = 16000
SETS = {
    "en_librispeech_clean": ("openslr/librispeech_asr", "clean", "en", "text"),
    "en_librispeech_other": ("openslr/librispeech_asr", "other", "en", "text"),   # harder accents / recording conditions
    "en_fleurs": ("google/fleurs", "en_us", "en", "transcription"),
    "zh_fleurs": ("google/fleurs", "cmn_hans_cn", "zh", "transcription"),
    "zh_aishell": ("shenyunhang/AISHELL-1", None, "zh", None),   # Apache-2.0; 20 test speakers, per-file wavs
}
AISHELL_REV = "2724409d538167445e43ebf846990319f12a1cbf"
_PINS = json.loads((ROOT / "eval" / "suites" / "pins.json").read_text())["datasets"]
REVISION = {p["repo"]: p["revision"] for p in _PINS.values()}   # dataset revisions pinned in eval/suites/pins.json


def norm_zh(t: str) -> str:
    """Scoring only: Chinese numerals -> Arabic (FLEURS/AISHELL refs use digits, ASR says 十五), then keep CJK + alnum."""
    import cn2an
    try:
        t = cn2an.transform(t, "cn2an")
    except Exception:  # noqa: BLE001 - odd numeral strings: score them as-is
        pass
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", t.lower())


def _en_numbers(t: str) -> str:
    """Scoring only: digits -> words (FLEURS refs write "25", the ASR says "twenty five"); 1100-2099 read as years."""
    from num2words import num2words

    def conv(m: re.Match) -> str:
        s = m.group(0)
        if "." in s:
            return num2words(float(s))
        n = int(s)
        return num2words(n, to="year") if len(s) == 4 and 1100 <= n <= 2099 and not 2000 <= n <= 2009 else num2words(n)
    return re.sub(r"\d+(?:\.\d+)?", conv, t.replace(",", ""))


def norm(t: str, lang: str, numbers: bool = False) -> str:
    t = t.lower()
    if lang == "zh":
        return norm_zh(t)
    if numbers:
        t = _en_numbers(t)
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", t).split())


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p = np.mean(x.astype(np.float64) ** 2) + 1e-9
    n = rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), x.shape)
    return np.clip(x + n, -32768, 32767).astype(np.int16)


def transcribe(recs, lang: str, pcm: np.ndarray, agc: bool | None = None) -> tuple[str, float]:
    """Same input path as the server: optional gain control (Settings.agc) in front of the streaming wrapper."""
    from vora.agc import Agc
    sess = AsrSession(recs, lang)
    gain = Agc() if (Settings().agc if agc is None else agc) else None
    x = np.concatenate([pcm, np.zeros(int(1.5 * SR), dtype=np.int16)])
    t0, finals = time.perf_counter(), []
    for i in range(0, len(x), 1600):
        frame = x[i:i + 1600].tobytes()
        finals += [e.text for e in sess.feed(gain.process(frame) if gain else frame) if e.kind == "final"]
    return ("" if lang == "zh" else " ").join(finals), time.perf_counter() - t0


def load(name: str, n: int, tag: str = ""):
    """Streams once from HF, then caches to data/eval_cache/<name><tag>.npz (HF streaming is flaky). The suites use a
    different tag so their bigger sets never change which 50 clips this script's own runs see."""
    cache = ROOT / "data" / "eval_cache" / f"{name}{tag}.npz"
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


def _stream_aishell(n: int):
    """n clips spread over the 20 AISHELL-1 test speakers (per-file download, ~150 KB each)."""
    import collections
    from huggingface_hub import HfApi, hf_hub_download
    files = [f for f in HfApi().list_repo_files("shenyunhang/AISHELL-1", repo_type="dataset", revision=AISHELL_REV) if "/wav/test/" in f]
    by = collections.defaultdict(list)
    for f in sorted(files):
        by[f.split("/")[-2]].append(f)
    per = -(-n // len(by))
    pick = [f for spk in sorted(by) for f in by[spk][:per]][:n]
    tr = Path(hf_hub_download("shenyunhang/AISHELL-1", "data_aishell/transcript/aishell_transcript_v0.8.txt", repo_type="dataset", revision=AISHELL_REV))
    text = {l.split(" ", 1)[0]: l.split(" ", 1)[1].replace(" ", "").strip() for l in tr.read_text(encoding="utf-8").splitlines() if " " in l}
    out = []
    for f in pick:
        p = hf_hub_download("shenyunhang/AISHELL-1", f, repo_type="dataset", revision=AISHELL_REV)
        x, sr = sf.read(p, dtype="float32")
        assert sr == SR, sr
        pcm = (x / (np.abs(x).max() + 1e-9) * 0.5 * 32767).astype(np.int16)
        out.append((text[Path(f).stem], pcm))
    return out


def _stream(name: str, n: int):
    if name == "zh_aishell":
        return _stream_aishell(n)
    repo, cfg, lang, col = SETS[name]
    ds = load_dataset(repo, cfg, split="test", streaming=True, revision=REVISION.get(repo)).cast_column("audio", Audio(decode=False))
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
