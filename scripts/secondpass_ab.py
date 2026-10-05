"""Measure-only (T5c/d): (1) offline second pass with sherpa-onnx-zipformer-small-en-2023-06-26 (int8, 27.6 MB; together with
the 43.6 MB streaming model the English ASR would be 71 MB, over the 50 MB limit, so it is never in the live path);
(2) the streaming model on quiet (-30 dB) and loud (+12 dB, clipped) input. Dev split, 60 clips per cell.
Usage: python scripts/secondpass_ab.py   (downloads the offline model into models/aux/ on first run)"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import eval_asr, suites  # noqa: E402
from scripts import eval_suite as E  # noqa: E402

REPO, REV = "csukuangfj/sherpa-onnx-zipformer-small-en-2023-06-26", None
D = ROOT / "models" / "aux" / "zipformer-small-en"


def offline():
    import sherpa_onnx
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, local_dir=D, allow_patterns=["*.int8.onnx", "tokens.txt"])
    f = lambda p: str(next(D.glob(p)))  # noqa: E731
    return sherpa_onnx.OfflineRecognizer.from_transducer(encoder=f("encoder*.int8.onnx"), decoder=f("decoder*.int8.onnx"),
                                                         joiner=f("joiner*.int8.onnx"), tokens=str(D / "tokens.txt"), num_threads=2)


def main() -> None:
    from vora.asr import load_recognizers
    from vora.config import Settings
    recs, off = load_recognizers(Settings()), offline()

    def second(x):
        s = off.create_stream()
        s.accept_waveform(16000, x.astype(np.float32) / 32768)
        off.decode_stream(s)
        return s.result.text

    for suite in ("librispeech_clean", "fleurs_en", "minds14_en_us"):
        rows = E.pick(suites.load_manifest(suite), "dev", False, 60)
        refs = [r["text"] for r in rows]
        pcm = [suites.load_wav(suite, r) for r in rows]
        first = [eval_asr.transcribe(recs, "en", x)[0] for x in pcm]
        sec = [second(x) for x in pcm]
        quiet = [eval_asr.transcribe(recs, "en", E.apply_aug(x, "quiet", r["id"], [], None, []))[0] for x, r in zip(pcm, rows)]
        loud = [eval_asr.transcribe(recs, "en", E.apply_aug(x, "loud", r["id"], [], None, []))[0] for x, r in zip(pcm, rows)]
        w = lambda h: E.score_asr(refs, h, "en")["value"] * 100  # noqa: E731
        print(f"{suite:18s} streaming {w(first):5.1f}%  second-pass {w(sec):5.1f}%  quiet(-30dB) {w(quiet):5.1f}%  loud(+12dB) {w(loud):5.1f}%", flush=True)


if __name__ == "__main__":
    main()
