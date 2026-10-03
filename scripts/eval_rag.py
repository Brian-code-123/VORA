"""RAG relevance (top-3, off-topic rejection) and answer faithfulness (keyword check) on real components.
Usage: eval_rag.py"""
import json
import re
from pathlib import Path

from vora.config import ROOT, Settings
from vora.llm import Llm
from vora.rag.ingest import load_qa
from vora.rag.retriever import Retriever

def has_kw(answer: str, kws: list[str]) -> bool:
    """Any keyword in the answer, ignoring spaces and hyphens ("2gb" == "2 GB", "3-meter" == "3 meter")."""
    squash = lambda t: re.sub(r"[\s\-]+", "", t.lower())
    stem = lambda t: re.sub(r"(?<=[a-z]{3})(es|s|ed|d|ing)$", "", t)    # erases / erased / erase
    return any(squash(k) in squash(answer) or (k.isalpha() and len(k) > 4 and stem(squash(k)) in squash(answer)) for k in kws)


def load_faithfulness() -> list[dict]:
    """eval/faithfulness.jsonl: q, kw (any substring of the lowercased answer), neg (negation question), split."""
    return [json.loads(l) for l in (ROOT / "eval" / "faithfulness.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def load_blind(name: str = "rag_blind3") -> list[dict]:
    """Frozen self-authored sets (sha256 next to them). rag_blind (v1) and rag_blind2 were EXPOSED after their first
    measurement and are tuning sets now; rag_blind3 was written and frozen before the LLM experiments that followed.
    The G5 gate uses rag_blind3, read once with --final."""
    return [json.loads(l) for l in (ROOT / "eval" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def score_blind(r: Retriever, llm: Llm | None, name: str = "rag_blind3") -> dict:
    rows = []
    for x in load_blind(name):
        hits = r.search(x["q"])
        if x["kind"] == "offtopic":
            rows.append({**x, "ok": not hits, "ctx": [h.chunk_id for h in hits]})
            continue
        top3 = bool({h.chunk_id for h in hits} & set(x["chunk_id"]))
        ans = ("".join(llm.stream(x["q"], hits)).lower() if hits else "") if llm else ""
        rows.append({**x, "top3": top3, "answer": ans, "ctx": [h.chunk_id for h in hits],
                     "ok": (has_kw(ans, x["kw"]) if llm else top3)})
    ans_rows = [x for x in rows if x["kind"] != "offtopic"]
    acc = lambda rs: round(sum(x["ok"] for x in rs) / len(rs), 3) if rs else None
    return {"n": len(rows), "top3": round(sum(x["top3"] for x in ans_rows) / len(ans_rows), 3), "faith": acc(ans_rows),
            "faith_with_refusals": acc(rows), "negation_acc": acc([x for x in rows if x["kind"] == "neg"]),
            "offtopic_rejected": acc([x for x in rows if x["kind"] == "offtopic"]),
            "misses": [{"q": x["q"], "ctx": x["ctx"], "answer": x.get("answer", "")[:160]} for x in rows if not x["ok"]]}


def main() -> None:
    import sys
    out_path = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT / "results" / "rag.json"
    final = "--final" in sys.argv   # rag_blind3 is read once, in the final run (like the test split)
    s = Settings()
    r = Retriever(s)
    qa = load_qa(ROOT / "eval" / "rag_qa.jsonl")
    res = {}
    for split in ("dev", "heldout"):
        qs = [q for q in qa if q["chunk_id"] and q["split"] == split]
        ok = [bool({h.chunk_id for h in r.search(q["q"])} & set(q["chunk_id"])) for q in qs]
        res[f"top3_{split}"] = {"acc": round(sum(ok) / len(ok), 3), "n": len(ok),
                                "misses": [q["q"] for q, o in zip(qs, ok) if not o]}
    off = [q["q"] for q in qa if not q["chunk_id"]]
    res["offtopic_rejected"] = {"rate": round(sum(r.search(q) == [] for q in off) / len(off), 3), "n": len(off)}
    llm = Llm(s)
    rows = []
    for item in load_faithfulness():
        ans = "".join(llm.stream(item["q"], r.search(item["q"]))).lower()
        rows.append({"q": item["q"], "answer": ans, "ok": has_kw(ans, item["kw"]), "neg": item["neg"], "split": item["split"]})
    acc = lambda rs: round(sum(x["ok"] for x in rs) / len(rs), 3) if rs else None
    res["faithfulness"] = {"acc": acc(rows), "n": len(rows), "negation_acc": acc([x for x in rows if x["neg"]]),
                           "heldout_acc": acc([x for x in rows if x["split"] == "heldout"]), "rows": rows}
    res["tune2"] = score_blind(r, llm, "rag_blind")      # exposed sets: report, never claim as held-out
    res["tune3"] = score_blind(r, llm, "rag_blind2")
    if final:
        blind = score_blind(r, llm, "rag_blind3")
        res["blind"] = blind
        n_all = len(rows) + blind["n"]
        ok_all = sum(x["ok"] for x in rows) + round(blind["faith_with_refusals"] * blind["n"])
        res["faithfulness_all"] = {"acc": round(ok_all / n_all, 3), "n": n_all, "misses_allowed_for_95pct": n_all // 20}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print({k: (v if k.startswith("top3") or k.startswith("off") or k in ("blind", "tune2", "tune3", "faithfulness_all") else {kk: v[kk] for kk in ("acc", "n", "negation_acc", "heldout_acc") if kk in v}) for k, v in res.items()})
    for x in rows:
        if not x["ok"]:
            print("  UNFAITHFUL:", x["q"], "->", x["answer"])


if __name__ == "__main__":
    main()
