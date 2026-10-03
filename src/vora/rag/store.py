import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from vora.guard import split_sentences

_LINE = re.compile(r"^\[([A-Z]\d+)\]\s+(.*\S)\s*$")
LANG_MODELS = {"zh": "BAAI/bge-small-zh-v1.5", "en": "BAAI/bge-small-en-v1.5"}   # one embedder per language (the zh model is weak on English)
EMBED_MODEL = LANG_MODELS["zh"]
_CJK = re.compile(r"[一-鿿]")


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str
    score: float
    focus: str = ""    # the best-matching sentence of the chunk (with the chunk title), set by the retriever


@dataclass
class Lane:
    """Everything one language needs: chunk index + sentence passages."""
    index: "faiss.Index"
    pos: np.ndarray                 # positions of this language's chunks in the global chunk list
    pvec: np.ndarray = field(default_factory=lambda: np.zeros((0, 1), np.float32))   # normalised passage embeddings
    pchunk: np.ndarray = field(default_factory=lambda: np.zeros(0, int))              # chunk position of each passage
    psent: list = field(default_factory=list)                                         # text a model should read for each passage


_TITLE = re.compile(r"^([^:：.。!?！？]{2,40})([:：])")


def split_passages(text: str) -> list[tuple[str, str]]:
    """[(sentence, embedding text)]: the first sentence already carries the "Title:", later ones get it prepended so a
    sentence like "The M100 warranty is 1 year." is still about warranties when it is embedded and when it is read."""
    sents = [x.strip() for x in split_sentences(text) if x.strip()]
    m = _TITLE.match(text.strip())
    title = m.group(1).strip() if m and len(m.group(1).split()) <= 6 else ""
    sep = (m.group(2) + (" " if m.group(2) == ":" else "")) if title else ""
    return [(s, s if j == 0 or not title else f"{title}{sep}{s}") for j, s in enumerate(sents)]


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


def save(index_dir: Path, chunks: list[dict], vecs: dict[str, np.ndarray], meta: dict, passages: dict | None = None) -> None:
    """vecs[lang] = normalised embeddings of the chunks whose lang == lang, in chunk order.
    passages[lang] = (normalised passage embeddings, [{"chunk": global chunk position, "text": text to read}])."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    for old in list(index_dir.glob("faiss*.index")) + list(index_dir.glob("passages_*")):
        old.unlink()
    for lang, v in vecs.items():
        idx = faiss.IndexFlatIP(v.shape[1])
        idx.add(v.astype(np.float32))
        faiss.write_index(idx, str(index_dir / f"faiss_{lang}.index"))
    for lang, (pv, rows) in (passages or {}).items():
        np.save(index_dir / f"passages_{lang}.npy", pv.astype(np.float32))
        (index_dir / f"passages_{lang}.json").write_text(json.dumps(rows, ensure_ascii=False))
    (index_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    (index_dir / "meta.json").write_text(json.dumps(meta))


def exists(index_dir: Path) -> bool:
    d = Path(index_dir)
    return (d / "chunks.json").exists() and any(d.glob("faiss_*.index"))


def load(index_dir: Path):
    """-> ({lang: Lane}, chunks, meta)"""
    d = Path(index_dir)
    if not exists(d):
        raise FileNotFoundError(f"no index in {d}: run python -m vora.rag.ingest (older single-index layouts must be rebuilt)")
    chunks = json.loads((d / "chunks.json").read_text())
    lanes = {}
    for f in sorted(d.glob("faiss_*.index")):
        lang = f.stem.split("_", 1)[1]
        lane = Lane(faiss.read_index(str(f)), np.array([i for i, c in enumerate(chunks) if c["lang"] == lang]))
        if (d / f"passages_{lang}.npy").exists():
            rows = json.loads((d / f"passages_{lang}.json").read_text())
            lane.pvec = np.load(d / f"passages_{lang}.npy")
            lane.pchunk = np.array([r["chunk"] for r in rows])
            lane.psent = [r["text"] for r in rows]
        lanes[lang] = lane
    return lanes, chunks, json.loads((d / "meta.json").read_text())
