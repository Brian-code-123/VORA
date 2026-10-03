import json
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

_LINE = re.compile(r"^\[([A-Z]\d+)\]\s+(.*\S)\s*$")
LANG_MODELS = {"zh": "BAAI/bge-small-zh-v1.5", "en": "BAAI/bge-small-en-v1.5"}   # one embedder per language (the zh model is weak on English)
EMBED_MODEL = LANG_MODELS["zh"]
_CJK = re.compile(r"[一-鿿]")


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str
    score: float


def lang_of(text: str) -> str:
    """zh when the text holds at least two CJK characters ("X200 拾音范围", "m100能连5g的wifi吗"), else en."""
    return "zh" if len(_CJK.findall(text)) >= 2 else "en"


def parse_kb(kb_dir: Path) -> list[dict]:
    """Chunks are lines `[ID] text` in kb/*.md."""
    out = []
    for f in sorted(Path(kb_dir).glob("*.md")):
        for line in f.read_text(encoding="utf-8").splitlines():
            m = _LINE.match(line)
            if m:
                out.append({"id": m.group(1), "text": m.group(2)})
    return out


def save(index_dir: Path, chunks: list[dict], vecs: dict[str, np.ndarray], meta: dict) -> None:
    """vecs[lang] = normalised embeddings of the chunks whose lang == lang, in chunk order."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    for old in index_dir.glob("faiss*.index"):
        old.unlink()
    for lang, v in vecs.items():
        idx = faiss.IndexFlatIP(v.shape[1])
        idx.add(v.astype(np.float32))
        faiss.write_index(idx, str(index_dir / f"faiss_{lang}.index"))
    (index_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    (index_dir / "meta.json").write_text(json.dumps(meta))


def exists(index_dir: Path) -> bool:
    d = Path(index_dir)
    return (d / "chunks.json").exists() and any(d.glob("faiss_*.index"))


def load(index_dir: Path):
    """-> ({lang: (faiss index, positions of that lang's chunks in `chunks`)}, chunks, meta)"""
    d = Path(index_dir)
    if not exists(d):
        raise FileNotFoundError(f"no index in {d}: run python -m vora.rag.ingest (older single-index layouts must be rebuilt)")
    chunks = json.loads((d / "chunks.json").read_text())
    lanes = {}
    for f in sorted(d.glob("faiss_*.index")):
        lang = f.stem.split("_", 1)[1]
        lanes[lang] = (faiss.read_index(str(f)), np.array([i for i, c in enumerate(chunks) if c["lang"] == lang]))
    return lanes, chunks, json.loads((d / "meta.json").read_text())
