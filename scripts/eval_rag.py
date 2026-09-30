"""RAG relevance (top-3, off-topic rejection) and answer faithfulness (keyword check) on real components.
Usage: eval_rag.py"""
import json

from vora.config import ROOT, Settings
from vora.llm import Llm
from vora.rag.ingest import load_qa
from vora.rag.retriever import Retriever

# question -> any of these substrings (lowercase) must appear in the spoken answer
FAITHFUL = {
    "how long is the warranty on the vora x200": ["2 years", "two years", "2 year"],
    "what is the wake word": ["hey vora"],
    "what does the red light mean": ["mute"],
    "how do i factory reset it": ["10 seconds", "reset button", "10 sec"],
    "how do i contact support": ["support@vora.example"],
    "does it understand cantonese": ["not support", "not supported", "no,", "cannot", "doesn't", "does not"],
    "who pays for return shipping": ["buyer"],
    "how much does the x200 cost": ["199"],
    "what temperature can it operate in": ["40"],
    "what power adapter does the x200 use": ["12 v", "12v"],
    "how many microphones does the x200 have": ["4", "four"],
    "how loud is the speaker": ["10 w", "10w", "10 watt"],
    "which file formats can i upload for the knowledge base": ["markdown", "txt", "pdf"],
    "how often does the firmware update": ["month"],
    "can i import my own documents": ["yes", "import", "document"],
    "X200保修期多久": ["2 年", "2年", "两年"],
    "怎么恢复出厂设置": ["10 秒", "10秒", "复位"],
    "唤醒词是什么": ["hey vora"],
    "支持哪些语言": ["普通话", "英语"],
    "知识库最大能导入多大": ["1 gb", "1gb"],
}


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
    for q, kws in FAITHFUL.items():
        ans = "".join(llm.stream(q, r.search(q))).lower()
        rows.append({"q": q, "answer": ans, "ok": any(k in ans for k in kws)})
    res["faithfulness"] = {"acc": round(sum(x["ok"] for x in rows) / len(rows), 3), "n": len(rows), "rows": rows}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "rag.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print({k: (v if k.startswith("top3") or k.startswith("off") else {"acc": v["acc"], "n": v["n"]}) for k, v in res.items()})
    for x in rows:
        if not x["ok"]:
            print("  UNFAITHFUL:", x["q"], "->", x["answer"])


if __name__ == "__main__":
    main()
