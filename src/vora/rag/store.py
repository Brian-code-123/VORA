import json
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

_LINE = re.compile(r"^\[([A-Z]\d+)\]\s+(.*\S)\s*$")
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str
    score: float


def parse_kb(kb_dir: Path) -> list[dict]:
    """Chunks are lines `[ID] text` in kb/*.md."""
    out = []
    for f in sorted(Path(kb_dir).glob("*.md")):
        for line in f.read_text(encoding="utf-8").splitlines():
            m = _LINE.match(line)
            if m:
                out.append({"id": m.group(1), "text": m.group(2)})
    return out


def save(index_dir: Path, chunks: list[dict], vecs: np.ndarray, meta: dict) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs.astype(np.float32))
    faiss.write_index(idx, str(index_dir / "faiss.index"))
    (index_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    (index_dir / "meta.json").write_text(json.dumps(meta))


def load(index_dir: Path):
    idx = faiss.read_index(str(Path(index_dir) / "faiss.index"))
    chunks = json.loads((Path(index_dir) / "chunks.json").read_text())
    meta = json.loads((Path(index_dir) / "meta.json").read_text())
    return idx, chunks, meta
