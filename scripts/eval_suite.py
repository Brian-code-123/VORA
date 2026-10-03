"""Real-speech evaluation on the public suites (scripts/suites.py): ASR error rate, retrieval top-1/top-3 against the
MInDS-14 intent (on the ASR text AND on the reference text, to isolate ASR damage), faithfulness, off-topic false-accept
and real-time end-to-end latency split into answered / refused turns.
The test split is read once: --split test needs --final. Results are written only with VORA_WRITE_RESULTS=1.
Usage: python scripts/eval_suite.py --suite minds14_en_us [--aug ambient@10] [--split dev] [--kb bank] [--faith] [--latency]
       python scripts/eval_suite.py --audio-dir DIR --latency        (your own wavs, optional DIR/texts.jsonl)"""
import argparse
import asyncio
import hashlib
import json
import re
import sys
from math import gcd, sqrt
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts import augment, suites  # noqa: E402

SR = 16000
BANK_KB, BANK_INDEX = ROOT / "eval_kb" / "bank", ROOT / "index_bank"
OFFTOPIC_SUITES = {"en": ("librispeech_clean", "fleurs_en"), "zh": ("aishell", "fleurs_zh")}


# ---------- statistics ----------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (0.0 if k == 0 else max(0.0, (c - h) / d), 1.0 if k == n else min(1.0, (c + h) / d))


def wer_ci(errs: list[int], lens: list[int], seed: int = 0, iters: int = 1000) -> tuple[float, float]:
    """95% bootstrap interval of corpus error rate (errors / reference length), resampling whole clips."""
    if not errs:
        return (0.0, 1.0)
    e, l = np.array(errs, float), np.array(lens, float)
    idx = np.random.default_rng(seed).integers(0, len(e), (iters, len(e)))
    v = e[idx].sum(1) / np.maximum(l[idx].sum(1), 1)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(np.floor(k)), int(np.ceil(k))
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# ---------- split discipline ----------
def pick(rows: list[dict], split: str, final: bool, n: int = 0) -> list[dict]:
    """The test split is for the final run only; a smaller n is a deterministic hash-ordered subset."""
    if split == "test" and not final:
        raise PermissionError("the test split is read once, in the final run: pass --final")
    out = sorted((r for r in rows if r["split"] == split), key=lambda r: hashlib.md5(r["id"].encode()).hexdigest())
    return out[:n] if n else out


# ---------- augmentation ----------
_AUG = re.compile(r"^(?:(clean|quiet|loud|codec)|(ambient|babble|reverb)@(\d+(?:\.\d+)?))$")


def parse_aug(spec: str) -> tuple[str, float | None]:
    m = _AUG.match(spec)
    if not m:
        raise ValueError(f"unknown augmentation {spec!r} (clean, quiet, loud, codec, ambient@dB, babble@dB, reverb@rt60)")
    return (m.group(1), None) if m.group(1) else (m.group(2), float(m.group(3)))


def apply_aug(pcm: np.ndarray, spec: str, row_id: str, pool_ids: list[str], load_other, noise: list[np.ndarray]) -> np.ndarray:
    kind, arg = parse_aug(spec)
    if kind == "clean":
        return pcm
    rng = np.random.default_rng(int(hashlib.md5(f"{row_id}|{spec}".encode()).hexdigest()[:8], 16))
    if kind == "ambient":
        if not noise:
            raise ValueError("no noise clips: python scripts/suites.py build ambient_noise")
        return augment.mix_snr(pcm, noise[int(rng.integers(len(noise)))], arg, rng)
    if kind == "babble":
        clips = [load_other(i) for i in augment.others(pool_ids, row_id, 4, rng)]
        return augment.mix_snr(pcm, augment.babble(clips, len(pcm), rng), arg, rng)
    if kind == "reverb":
        return augment.reverb(pcm, arg, SR, rng)
    if kind == "quiet":
        return augment.gain(pcm, -30)
    if kind == "loud":
        return augment.gain(pcm, 12)
    return augment.telephone(pcm, SR)   # codec


# ---------- scoring ----------
def score_asr(refs: list[str], hyps: list[str], lang: str) -> dict:
    import jiwer
    from scripts.eval_asr import norm
    errs, lens = [], []
    for ref, hyp in zip(refs, hyps):
        r, h = norm(ref, lang), norm(hyp, lang)
        unit = list if lang == "zh" else str.split
        n = len(unit(r))
        if n == 0:
            continue
        if not h:
            errs.append(n)
        else:
            o = (jiwer.process_characters if lang == "zh" else jiwer.process_words)(r, h)
            errs.append(o.substitutions + o.deletions + o.insertions)
        lens.append(n)
    tot = sum(lens)
    lo, hi = wer_ci(errs, lens)
    return {"metric": "cer" if lang == "zh" else "wer", "value": (sum(errs) / tot) if tot else None, "errors": sum(errs),
            "ref_len": tot, "n": len(lens), "ci": [lo, hi]}


def faith_check(answer: str, kws: list[str]) -> bool:
    """Any keyword in the answer, ignoring case, spaces and hyphens ("2gb" == "2 GB")."""
    squash = lambda t: re.sub(r"[\s\-]+", "", t.lower())
    stem = lambda t: re.sub(r"(?<=[a-z]{3})(es|s|ed|d|ing)$", "", t)    # erases / erased / erase
    return any(squash(k) in squash(answer) or (k.isalpha() and len(k) > 4 and stem(squash(k)) in squash(answer)) for k in kws)


def _retrieve(retriever, text: str, ids: set[str]) -> tuple[bool, bool, list]:
    hits = retriever.search(text) if text.strip() else []
    return bool(hits) and hits[0].chunk_id in ids, any(h.chunk_id in ids for h in hits[:3]), hits


def offtopic_false_accept(rows: list[dict], retriever) -> float:
    """Share of unrelated real sentences (read speech) that still retrieve something."""
    return sum(bool(retriever.search(r["text"])) for r in rows) / len(rows) if rows else 0.0


def evaluate(rows: list[dict], load_pcm, transcribe_fn, retriever=None, intents: dict | None = None, llm=None,
             aug: str = "clean", noise: list[np.ndarray] | None = None, unsure: dict | None = None) -> dict:
    by_id = {r["id"]: r for r in rows}
    ids = list(by_id)
    refs, hyps, per, skipped = [], [], [], 0
    rt = {"n": 0, "top1": 0, "top3": 0, "refused": 0, "top1_ref": 0, "top3_ref": 0}
    fa = {"n": 0, "ok": 0, "ok_ref": 0}
    for r in rows:
        if not r["text"].strip():
            skipped += 1
            continue
        pcm = apply_aug(load_pcm(r), aug, r["id"], ids, lambda i: load_pcm(by_id[i]), noise or [])
        hyp = transcribe_fn(r, pcm)
        refs.append(r["text"]); hyps.append(hyp)
        rec = {"id": r["id"], "ref": r["text"], "hyp": hyp}
        if retriever is not None and intents and r.get("intent") in intents:
            want = set(intents[r["intent"]]["chunk_ids"])
            t1, t3, hits = _retrieve(retriever, hyp, want)
            t1r, t3r, hits_ref = _retrieve(retriever, r["text"], want)
            rt["n"] += 1; rt["top1"] += t1; rt["top3"] += t3; rt["refused"] += not hits; rt["top1_ref"] += t1r; rt["top3_ref"] += t3r
            rec.update(top3=t3, top3_ref=t3r, refused=not hits, ctx=[h.chunk_id for h in hits])
            if llm is not None:
                kws = intents[r["intent"]]["kw"][r["lang"]]
                ans = "".join(llm.stream(hyp, hits)) if hits else (unsure or {}).get(r["lang"], "")
                ans_ref = "".join(llm.stream(r["text"], hits_ref)) if hits_ref else ""
                fa["n"] += 1; fa["ok"] += faith_check(ans, kws); fa["ok_ref"] += faith_check(ans_ref, kws)
                rec.update(answer=ans, faith=faith_check(ans, kws))
        per.append(rec)
    out = {"n": len(refs), "n_skipped_empty_ref": skipped, "asr": score_asr(refs, hyps, rows[0]["lang"]) if rows else None, "rows": per}
    if rt["n"]:
        n = rt["n"]
        out["retrieval"] = {**rt, **{f"{k}_ci": list(wilson(rt[k], n)) for k in ("top1", "top3", "top1_ref", "top3_ref")},
                            **{k: rt[k] / n for k in ("top1", "top3", "top1_ref", "top3_ref")}, "refused_rate": rt["refused"] / n}
    if fa["n"]:
        out["faith"] = {"n": fa["n"], "acc": fa["ok"] / fa["n"], "ci": list(wilson(fa["ok"], fa["n"])), "acc_ref": fa["ok_ref"] / fa["n"]}
    return out


# ---------- latency + cross-check ----------
def latency_summary(turns: list[dict | None]) -> dict:
    got = [t for t in turns if t and t.get("first_content_audio_ms") is not None]
    ans = [t["first_content_audio_ms"] for t in got if t.get("context")]
    ref = [t["first_content_audio_ms"] for t in got if not t.get("context")]
    both = lambda xs: {"p50": round(_pct(xs, 50), 1), "p90": round(_pct(xs, 90), 1), "n": len(xs)} if xs else None
    out = {"n_answered": len(ans), "n_refused": len(ref), "n_no_response": len(turns) - len(got), "answered": both(ans), "all": both(ans + ref),
           "refusal_rate": len(ref) / len(got) if got else None}
    adj = [t["first_content_audio_ms"] - t["silero_gap_ms"] for t in got if t.get("context") and t.get("silero_gap_ms") is not None]
    if adj:   # same turns, speech_end taken from silero instead of the pipeline's RMS rule
        out["answered_vs_silero"] = both(adj)
    return out


def gap_report(gaps_ms: list[float]) -> dict:
    over = sum(g > 150 for g in gaps_ms)
    return {"n": len(gaps_ms), "median_ms": float(np.median(gaps_ms)) if gaps_ms else None, "max_ms": max(gaps_ms) if gaps_ms else None,
            "n_over_150ms": over, "flagged": bool(gaps_ms) and over / len(gaps_ms) > 0.2}


def _speech_end_gaps(rows: list[dict], load_pcm, limit: int = 40) -> dict | None:
    from scripts import refvad
    if not refvad.SILERO_PATH.exists():
        return None
    gaps = []
    for r in rows[:limit]:
        x = np.concatenate([np.zeros(SR // 2, np.int16), load_pcm(r), np.zeros(SR, np.int16)])
        gaps.append(abs(refvad.speech_end(x) - refvad.pipeline_speech_end(x)) * 1000)
    return gap_report(gaps)


# ---------- local wavs ----------
def rows_from_dir(d: Path, default_lang: str = "en") -> list[dict]:
    texts = {}
    tj = Path(d) / "texts.jsonl"
    if tj.exists():
        for line in tj.read_text(encoding="utf-8").splitlines():
            if line.strip():
                j = json.loads(line)
                texts[j["file"]] = j
    rows = []
    for w in sorted(Path(d).glob("*.wav")):
        t = texts.get(w.name, {})
        rows.append(suites.make_row(id=w.stem, wav=w.name, text=t.get("text", ""), lang=t.get("lang", default_lang), scenario="own",
                                    accent="", intent=None, source="local recording", licence="own recording, not redistributed",
                                    orig_sr=sf.info(w).samplerate))
    return rows


def load_dir_pcm(d: Path, row: dict) -> np.ndarray:
    x, sr = sf.read(Path(d) / row["wav"], dtype="float32")
    x = x[:, 0] if x.ndim > 1 else x
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return np.clip(np.rint(x * 32768), -32768, 32767).astype(np.int16)


# ---------- plumbing ----------
def finish(res: dict, qm: dict, out_dir: Path | None = None) -> dict:
    from scripts.bench import host
    from vora.hostcheck import write_result
    res.update(quiet=bool(qm["quiet"]), max_other_cpu_pct=qm["max_other_cpu_pct"], p90_other_cpu_pct=qm["p90_other_cpu_pct"], host=host())
    d = Path(out_dir) if out_dir else ROOT / "results" / "suites"
    write_result(d / f"{res['suite']}__{res['aug']}__{res['split']}.json", res)
    return res


async def _latency_turns(models, settings, items: list[tuple[str, np.ndarray]]) -> list[dict | None]:
    from scripts.bench import one_turn
    out = []
    for i, (lang, pcm) in enumerate(items):
        m = await one_turn(models, settings, lang, pcm)
        out.append(m)
        print(f"  latency {i + 1}/{len(items)} {lang} heard={(m or {}).get('heard')!r} answered={bool((m or {}).get('context'))} "
              f"first_content={(m or {}).get('first_content_audio_ms')}", flush=True)
    return out


_CACHE: dict = {}


def _models(kb: str, faith: bool):
    """Recognizers / retriever / LLM are loaded once per process (a matrix run reuses them for every cell)."""
    from vora.config import Settings
    s = Settings(kb_dir=BANK_KB, index_dir=BANK_INDEX) if kb == "bank" else Settings()
    if "recs" not in _CACHE:
        from vora.asr import load_recognizers
        _CACHE["recs"] = load_recognizers(Settings())
    if kb == "bank" and "retriever" not in _CACHE:
        from vora.rag.retriever import Retriever
        _CACHE["retriever"] = Retriever(s)
    if kb == "bank" and faith and "llm" not in _CACHE:
        from vora.llm import Llm
        _CACHE["llm"] = Llm(s)
    intents = json.loads((BANK_KB / "intents.json").read_text(encoding="utf-8")) if kb == "bank" else None
    return s, _CACHE["recs"], _CACHE.get("retriever") if kb == "bank" else None, _CACHE.get("llm") if (kb == "bank" and faith) else None, intents


def run(suite: str, aug: str = "clean", split: str = "dev", final: bool = False, n: int = 0, kb: str = "none",
        faith: bool = False, latency: bool = False, audio_dir: Path | None = None, force: bool = False) -> dict:
    if audio_dir is None and suite not in suites.ALL:
        raise KeyError(f"unknown suite: {suite}")
    from vora.config import Settings
    from vora.hostcheck import QuietMonitor, perf_skip_reason
    if latency and not force and (reason := perf_skip_reason()):
        raise SystemExit(f"refusing to measure latency: {reason}. --force records it with quiet=false.")
    if audio_dir is not None:
        rows, load_pcm, split = rows_from_dir(audio_dir), (lambda r: load_dir_pcm(audio_dir, r)), "all"
        pool_rows = rows
    else:
        pool_rows = suites.load_manifest(suite)
        rows = pick(pool_rows, split, final, n)
        load_pcm = lambda r: suites.load_wav(suite, r)  # noqa: E731
    noise = []
    if aug.startswith("ambient"):
        noise = [suites.load_wav(suites.NOISE, r) for r in suites.load_manifest(suites.NOISE)]
    from scripts import eval_asr
    s, recs, retriever, llm, intents = _models(kb, faith)
    mon = QuietMonitor()
    mon.start()
    res = {"suite": suite if audio_dir is None else f"dir:{Path(audio_dir).name}", "aug": aug, "split": split, "kb": kb, **evaluate(
        rows, load_pcm, lambda r, x: eval_asr.transcribe(recs, r["lang"], x)[0], retriever, intents, llm, aug, noise,
        unsure={"en": "I'm not sure", "zh": "我不确定"})}
    if retriever is not None and rows:
        lang = rows[0]["lang"]
        off = [r for n_ in OFFTOPIC_SUITES[lang] for r in [x for x in suites.load_manifest(n_) if x["split"] == split][:100]]
        res["offtopic_false_accept"] = {"rate": offtopic_false_accept(off, retriever), "n": len(off)}
    res["speech_end_gap"] = _speech_end_gaps(rows, load_pcm)
    if latency:
        ids = None if audio_dir else json.loads((suites.PINS).read_text())["latency_ids"].get(rows[0]["lang"], {}).get(split)
        lat_rows = rows if ids is None else [r for r in pool_rows if r["id"] in set(ids)]
        by_id = {r["id"]: r for r in pool_rows}
        items = [(r["lang"], apply_aug(load_pcm(r), aug, r["id"], list(by_id), lambda i: load_pcm(by_id[i]), noise)) for r in lat_rows]
        from vora.server import Models
        turns = asyncio.run(_latency_turns(Models.load(s), s, items))
        from scripts import refvad
        if refvad.SILERO_PATH.exists():
            for t, (_, x) in zip(turns, items):
                if t is not None:   # one_turn pads 0.5 s of silence in front; both rules see the same padded signal
                    y = np.concatenate([np.zeros(SR // 2, np.int16), x])
                    t["silero_gap_ms"] = round((refvad.speech_end(y) - refvad.pipeline_speech_end(y)) * 1000, 1)
        res["latency"] = latency_summary(turns)
    return finish(res, mon.stop())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="")
    ap.add_argument("--audio-dir", type=Path)
    ap.add_argument("--aug", default="clean")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true", help="allow reading the test split (final run only)")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--kb", default="none", choices=["none", "bank"])
    ap.add_argument("--faith", action="store_true")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--force", action="store_true", help="measure latency on a busy host (recorded quiet=false)")
    a = ap.parse_args(argv)
    res = run(a.suite, a.aug, a.split, a.final, a.n, a.kb, a.faith, a.latency, a.audio_dir, a.force)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
