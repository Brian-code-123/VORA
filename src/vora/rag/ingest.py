"""python -m vora.rag.ingest  -> builds index/ and calibrates min_score on the dev split."""
import json
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from vora.config import ROOT, Settings
from vora.rag import store


def ingest(kb_dir: Path, index_dir: Path) -> None:
    chunks = store.parse_kb(kb_dir)
    emb = TextEmbedding(store.EMBED_MODEL, threads=2)
    vecs = np.array(list(emb.embed([c["text"] for c in chunks])), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    store.save(index_dir, chunks, vecs, {"embed_model": store.EMBED_MODEL, "n": len(chunks)})


def load_qa(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def calibrate(settings: Settings, qa_path: Path) -> dict:
    """Grid-search min_score on dev: maximise top-3 accuracy subject to rejecting >=90% of off-topic queries."""
    from vora.rag.retriever import Retriever
    r = Retriever(settings)
    r.min_score = 0.0
    dev = [q for q in load_qa(qa_path) if q["split"] == "dev"]
    ans = [q for q in dev if q["chunk_id"]]
    off = [q for q in dev if not q["chunk_id"]]
    hit = {q["q"]: {h.chunk_id for h in r.search(q["q"])} & set(q["chunk_id"]) != set() for q in ans}
    ans_score = {q["q"]: r.best_score(q["q"]) for q in ans}
    off_score = [r.best_score(q["q"]) for q in off]
    best = None
    for th in np.arange(0.2, 0.9, 0.01):
        rej = np.mean([s < th for s in off_score])
        acc = np.mean([hit[q] and ans_score[q] >= th for q in hit])
        if rej >= 0.9 and (best is None or acc > best[1]):
            best = (float(round(th, 2)), float(acc), float(rej))
    best = best or (float(max(off_score)) + 0.01, 0.0, 1.0)
    meta_path = settings.index_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(min_score=best[0], dev_top3_at_threshold=best[1], offtopic_rejected=best[2])
    meta_path.write_text(json.dumps(meta))
    return meta


if __name__ == "__main__":
    s = Settings()
    ingest(s.kb_dir, s.index_dir)
    print(calibrate(s, ROOT / "eval" / "rag_qa.jsonl"))
