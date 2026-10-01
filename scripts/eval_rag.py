"""RAG relevance (top-3, off-topic rejection) and answer faithfulness (keyword check) on real components.
Usage: eval_rag.py"""
import json

from vora.config import ROOT, Settings
from vora.llm import Llm
from vora.rag.ingest import load_qa
from vora.rag.retriever import Retriever

def load_faithfulness() -> list[dict]:
    """eval/faithfulness.jsonl: q, kw (any substring of the lowercased answer), neg (negation question), split."""
    return [json.loads(l) for l in (ROOT / "eval" / "faithfulness.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
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
        rows.append({"q": item["q"], "answer": ans, "ok": any(k.lower() in ans for k in item["kw"]), "neg": item["neg"], "split": item["split"]})
    acc = lambda rs: round(sum(x["ok"] for x in rs) / len(rs), 3) if rs else None
    res["faithfulness"] = {"acc": acc(rows), "n": len(rows), "negation_acc": acc([x for x in rows if x["neg"]]),
                           "heldout_acc": acc([x for x in rows if x["split"] == "heldout"]), "rows": rows}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "rag.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print({k: (v if k.startswith("top3") or k.startswith("off") else {kk: v[kk] for kk in ("acc", "n", "negation_acc", "heldout_acc") if kk in v}) for k, v in res.items()})
    for x in rows:
        if not x["ok"]:
            print("  UNFAITHFUL:", x["q"], "->", x["answer"])


if __name__ == "__main__":
    main()
