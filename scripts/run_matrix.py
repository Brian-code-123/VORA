"""Scene matrix: ASR error rate (+ retrieval / faithfulness on the MInDS-14 suites) for every suite x augmentation,
models loaded once. Usage: python scripts/run_matrix.py --split dev --n 100 --out results/suites_baseline.json
The test split needs --final (final run only)."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts import eval_suite  # noqa: E402

MINDS = ("minds14_en_us", "minds14_en_gb", "minds14_en_au", "minds14_zh")
NOISY = ("ambient@10", "ambient@5", "babble@10", "reverb@0.6", "quiet", "loud")
READ = {"librispeech_clean": ("ambient@10", "codec"), "librispeech_other": ("ambient@10",), "fleurs_en": ("ambient@10",),
        "aishell": ("ambient@10",), "fleurs_zh": ("ambient@10",)}


def cells() -> list[tuple[str, str, str]]:
    out = [(m, "clean", "bank") for m in MINDS]
    out += [(m, a, "bank") for m in ("minds14_en_us", "minds14_zh") for a in NOISY]
    for name, augs in READ.items():
        out += [(name, "clean", "none")] + [(name, a, "none") for a in augs]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--faith", action="store_true", help="run the LLM for faithfulness on the MInDS-14 clean cells")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, t0 = {}, time.time()
    for suite, aug, kb in cells():
        r = eval_suite.run(suite, aug, a.split, a.final, a.n, kb, faith=a.faith and aug == "clean" and kb == "bank",
                           out_dir=ROOT / "results" / "suites" / "matrix")   # keep results/suites/*.json (they hold the latency runs)
        r.pop("rows", None)
        out[f"{suite}|{aug}|{a.split}"] = r
        asr, rt = r["asr"], r.get("retrieval")
        print(f"{suite:18s} {aug:11s} n={r['n']:3d} {asr['metric']}={asr['value'] * 100:5.1f}%" + (
            f"  top3={rt['top3'] * 100:4.0f}% (ref {rt['top3_ref'] * 100:4.0f}%) refused={rt['refused_rate'] * 100:3.0f}%" if rt else "") + (
            f"  faith={r['faith']['acc'] * 100:4.0f}%" if r.get("faith") else "") + f"  [{time.time() - t0:.0f}s]", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
