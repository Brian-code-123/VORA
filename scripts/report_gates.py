"""Gate table from results/*.json: PASS / FAIL / UNVERIFIED. Usage: report_gates.py [--write results/gates_x.json]"""
import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from vora.hostcheck import is_quiet

ROOT = Path(__file__).resolve().parent.parent
BAD_LICENCE = re.compile(r"unknown|custom|non-?commercial", re.I)


@dataclass
class Gate:
    id: str
    name: str
    target: str
    measured: str
    ok: bool | None   # None = unverified
    quiet: bool


def _load(d: Path, name: str):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def evaluate(results_dir: Path) -> list[Gate]:
    d = Path(results_dir)
    bench, mem, asr, rag = (_load(d, f) for f in ("bench.json", "memory.json", "asr.json", "rag.json"))
    conc, docker = _load(d, "concurrent.json"), _load(d, "docker.json")
    quiet = False
    if bench:
        h = bench.get("host", {})
        # bench.json records an instantaneous-CPU verdict; fall back to the load average for older results
        quiet = bool(bench["quiet"]) if "quiet" in bench else is_quiet(h.get("loadavg_1m", 99.0), h.get("cores"))
    gs: list[Gate] = []

    en = [r["total"] for r in (bench or {}).get("rows", []) if r.get("lang") == "en" and "total" in r]
    if en:
        p50, p90 = _pct(en, 50), _pct(en, 90)
        gs.append(Gate("G1", "end-to-end latency (en audio)", "p50 <=1500 ms, p90 <=1800 ms", f"p50 {p50:.0f} / p90 {p90:.0f} ms", p50 <= 1500 and p90 <= 1800, quiet))
    else:
        gs.append(Gate("G1", "end-to-end latency (en audio)", "p50 <=1500 ms, p90 <=1800 ms", "no data", None, quiet))

    tf = _load(d, "tts_first_chunk.json")
    if tf:   # dedicated test: text of the real first chunk -> first PCM piece (the bench metric also waits for LLM tokens)
        worst = max(tf["en_p95_ms"], tf["zh_p95_ms"])
        gs.append(Gate("G2", "TTS first chunk", "p95 <=200 ms", f"en p95 {tf['en_p95_ms']:.0f} / zh {tf['zh_p95_ms']:.0f} ms", worst <= 200, bool(tf.get("quiet"))))
    else:
        t = ((bench or {}).get("latency_ms_oracle_text") or {}).get("tts_first_chunk")
        gs.append(Gate("G2", "TTS first chunk", "p95 <=200 ms", f"p95 {t['p95']:.0f} ms" if t else "no data", (t["p95"] <= 200) if t else None, quiet))

    gs.append(Gate("G3", "ASR+RAG+TTS memory", "<=500 MB (USS)", f"{mem['asr_rag_tts']:.0f} MB" if mem else "no data",
                   (mem["asr_rag_tts"] <= 500) if mem else None, True))   # USS does not depend on host load

    if asr:
        en_w = asr.get("en_librispeech_clean/clean", {}).get("value")
        zh_c = asr.get("zh_aishell/clean", {}).get("value")
        meas = f"en {en_w*100:.1f}%" if en_w is not None else "en n/a"
        meas += f", zh AISHELL {zh_c*100:.1f}%" if zh_c is not None else ", zh AISHELL missing"
        ok = None if (en_w is None or zh_c is None) else (en_w <= 0.15 and zh_c <= 0.15)
        if en_w is not None and en_w > 0.15:
            ok = False
        gs.append(Gate("G4", "ASR accuracy", "WER/CER <=15%", meas, ok, True))
    else:
        gs.append(Gate("G4", "ASR accuracy", "WER/CER <=15%", "no data", None, True))

    if rag:
        top = min(rag["top3_dev"]["acc"], rag["top3_heldout"]["acc"])
        fa = rag["faithfulness"]["acc"]
        gs.append(Gate("G5", "RAG top-3 + faithfulness", "top-3 >=80%, faithfulness >=95%", f"top-3 {top*100:.0f}%, faithful {fa*100:.0f}%", top >= 0.8 and fa >= 0.95, True))
    else:
        gs.append(Gate("G5", "RAG top-3 + faithfulness", "top-3 >=80%, faithfulness >=95%", "no data", None, True))

    gs.append(Gate("G6", "2 concurrent users", "each p50 <=2x single", json.dumps(conc.get("summary")) if conc else "not measured",
                   conc.get("ok") if conc else None, bool(conc.get("quiet")) if conc else False))
    gs.append(Gate("G7", "Docker + Pi proof", "image builds, health 200, Pi run", json.dumps(docker.get("summary")) if docker else "not built / no Pi",
                   docker.get("ok") if docker else None, True))

    man = (d.parent / "models" / "MANIFEST.json")
    if man.exists():
        lic = {k: v.get("license", "") for k, v in json.loads(man.read_text()).items()}
        bad = [k for k, v in lic.items() if BAD_LICENCE.search(v)]
        gs.append(Gate("G8", "permissive licences", "no custom/unknown", f"non-permissive: {', '.join(bad) or 'none'}", not bad, True))
    else:
        gs.append(Gate("G8", "permissive licences", "no custom/unknown", "no MANIFEST", None, True))
    return gs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write")
    ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args()
    gs = evaluate(Path(a.results))
    for g in gs:
        s = {True: "PASS", False: "FAIL", None: "UNVERIFIED"}[g.ok]
        print(f"{g.id} {s:10s} {g.name:34s} target {g.target:34s} measured {g.measured}{'' if g.quiet else '  [busy host]'}")
    if a.write:
        Path(a.write).write_text(json.dumps([asdict(g) for g in gs], indent=1))


if __name__ == "__main__":
    main()
