"""A/B of the gain control (Settings.agc) on the ASR input. Dev split, 60 clips per cell.
Usage: python scripts/agc_ab.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import eval_asr, suites  # noqa: E402
from scripts import eval_suite as E  # noqa: E402


def main() -> None:
    from vora.asr import load_recognizers
    from vora.config import Settings
    recs = load_recognizers(Settings())
    for suite, lang in (("librispeech_clean", "en"), ("fleurs_en", "en"), ("minds14_en_us", "en"), ("aishell", "zh"), ("minds14_zh", "zh")):
        rows = E.pick(suites.load_manifest(suite), "dev", False, 60)
        refs = [r["text"] for r in rows]
        out = []
        for aug in ("clean", "quiet", "loud"):
            for agc in (False, True):
                hyps = [eval_asr.transcribe(recs, lang, E.apply_aug(suites.load_wav(suite, r), aug, r["id"], [], None, []), agc=agc)[0] for r in rows]
                out.append(f"{aug}/{'agc' if agc else 'raw'} {E.score_asr(refs, hyps, lang)['value'] * 100:5.1f}")
        print(f"{suite:18s} " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
