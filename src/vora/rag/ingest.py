"""python -m vora.rag.ingest  -> builds index/ and calibrates min_score on the dev split."""
import json
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from vora.config import ROOT, Settings
from vora.rag import store


def ingest(kb_dir: Path, index_dir: Path) -> None:
    """One index per language, each with its own embedder (en chunks: bge-small-en, zh chunks: bge-small-zh). Every chunk is
    also split into sentence passages, embedded alongside, so a question can match the one sentence that answers it."""
    chunks = store.parse_kb(kb_dir)
    for c in chunks:
        c["lang"] = store.lang_of(c["text"])
    vecs, passages = {}, {}
    norm = lambda v: v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    for lang in sorted({c["lang"] for c in chunks}):
        emb = TextEmbedding(store.LANG_MODELS[lang], threads=2)
        mine = [(i, c) for i, c in enumerate(chunks) if c["lang"] == lang]
        vecs[lang] = norm(np.array(list(emb.embed([c["text"] for _, c in mine])), dtype=np.float32))
        prow = [(i, sent, text) for i, c in mine for sent, text in store.split_passages(c["text"])]
        pv = norm(np.array(list(emb.embed([t for _, _, t in prow])), dtype=np.float32))
        passages[lang] = (pv, [{"chunk": i, "text": t} for i, _, t in prow])
    store.save(index_dir, chunks, vecs, {"embed_models": {l: store.LANG_MODELS[l] for l in vecs}, "n": len(chunks)}, passages)


def load_qa(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def calibrate(settings: Settings, qa_path: Path | list[Path]) -> dict:
    """Per language, grid-search min_score on DEV questions: maximise top-3 accuracy subject to rejecting >=90% of
    off-topic queries. (One shared threshold was wrong: zh and en scores live on different scales.)"""
    from vora.rag.retriever import Retriever
    r = Retriever(settings)
    r.min_score = {lang: 0.0 for lang in r.lanes}
    paths = qa_path if isinstance(qa_path, list) else [qa_path]
    dev = [q for p in paths for q in load_qa(p) if q["split"] == "dev"]
    out = {"min_score": {}, "dev_top3_at_threshold": {}, "offtopic_rejected": {}, "n_dev": {}}
    for lang in r.lanes:
        qs = [q for q in dev if store.lang_of(q["q"]) == lang]
        ans, off = [q for q in qs if q["chunk_id"]], [q for q in qs if not q["chunk_id"]]
        if not ans or not off:      # nothing to calibrate against: keep the configured default
            out["min_score"][lang], out["n_dev"][lang] = settings.min_score, [len(ans), len(off)]
            continue
        hit = {q["q"]: {h.chunk_id for h in r.search(q["q"])} & set(q["chunk_id"]) != set() for q in ans}
        ans_score = {q["q"]: r.best_score(q["q"]) for q in ans}
        off_score = [r.best_score(q["q"]) for q in off]
        best = None
        for th in np.arange(0.2, 0.95, 0.01):
            rej = np.mean([s < th for s in off_score])
            acc = np.mean([hit[q] and ans_score[q] >= th for q in hit])
            if rej >= 0.9 and (best is None or acc > best[1]):
                best = (float(round(th, 2)), float(acc), float(rej))
        best = best or (float(max(off_score)) + 0.01, 0.0, 1.0)
        out["min_score"][lang], out["dev_top3_at_threshold"][lang], out["offtopic_rejected"][lang] = best
        out["n_dev"][lang] = [len(ans), len(off)]
    meta_path = settings.index_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(out)
    meta_path.write_text(json.dumps(meta))
    return meta


if __name__ == "__main__":
    s = Settings()
    ingest(s.kb_dir, s.index_dir)
    print(calibrate(s, [ROOT / "eval" / "rag_qa.jsonl", ROOT / "eval" / "offtopic_dev.jsonl"]))
