"""Index eval_kb/bank into index_bank/ and calibrate min_score on DEV data only:
MInDS-14 dev reference texts (answerable) + dev sentences of unrelated read speech (off-topic). The test split is untouched.
Usage: python scripts/build_bank_index.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts import suites  # noqa: E402
from vora.config import Settings  # noqa: E402
from vora.rag import ingest  # noqa: E402

BANK = ROOT / "eval_kb" / "bank"
INDEX = ROOT / "index_bank"
PER_SUITE = 100


def dev_qa() -> list[dict]:
    intents = json.loads((BANK / "intents.json").read_text(encoding="utf-8"))
    qa = []
    for name in suites.MINDS14:
        rows = [r for r in suites.load_manifest(name) if r["split"] == "dev"][:PER_SUITE]
        qa += [{"q": r["text"], "chunk_id": intents[r["intent"]]["chunk_ids"], "split": "dev"} for r in rows]
    for name in ("librispeech_clean", "fleurs_en", "aishell", "fleurs_zh"):
        qa += [{"q": r["text"], "chunk_id": [], "split": "dev"} for r in [r for r in suites.load_manifest(name) if r["split"] == "dev"][:50]]
    qa += [json.loads(l) for l in (ROOT / "eval" / "offtopic_dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]   # question-style off-topic
    return qa


def main() -> None:
    s = Settings(kb_dir=BANK, index_dir=INDEX)
    ingest.ingest(s.kb_dir, s.index_dir)
    qa = dev_qa()
    path = ROOT / "data" / "suites" / "bank_dev_qa.jsonl"
    path.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qa), encoding="utf-8")
    print({"n_answerable": sum(bool(q["chunk_id"]) for q in qa), "n_offtopic": sum(not q["chunk_id"] for q in qa)})
    print(ingest.calibrate(s, path))


if __name__ == "__main__":
    main()
