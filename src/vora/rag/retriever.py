import json
import re
from pathlib import Path

import jieba
import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

from vora.config import Settings
from vora.rag import store
from vora.rag.store import EMBED_MODEL, Hit

jieba.setLogLevel(60)
_LATIN_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\- ]*[A-Za-z0-9]|[A-Za-z0-9]")


def tokenize(text: str) -> list[str]:
    t = text.lower()
    toks = [w for w in jieba.lcut(t) if re.search(r"\w", w)]
    # "vora-x200" also indexed as "voraX200"-style compact token so codes match across spacing
    toks += [re.sub(r"[^a-z0-9]", "", w) for w in re.findall(r"[a-z]+[\- ]?[a-z]?[\- ]?\d+", t)]
    return toks


def _compact(s: str) -> str:
    return re.sub(r"[^a-z0-9一-鿿]", "", s.lower())


class Retriever:
    def __init__(self, settings: Settings, index_dir: Path | None = None):
        self.s = settings
        self.index, self.chunks, meta = store.load(index_dir or settings.index_dir)
        self.min_score = meta.get("min_score", settings.min_score)
        self.emb = TextEmbedding(EMBED_MODEL, threads=1)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        gpath = settings.kb_dir / "glossary.json"
        self.glossary = json.loads(gpath.read_text(encoding="utf-8")) if gpath.exists() else {}
        spath = settings.kb_dir / "synonyms.json"
        self.synonyms = json.loads(spath.read_text(encoding="utf-8")) if spath.exists() else {}
        self._variants = [(canon, _compact(v)) for canon, vs in self.glossary.items() for v in [canon, *vs]]

    def normalize_query(self, text: str) -> str:
        """Fuzzy-map ASR-mangled product terms ("vora x 200") to glossary canonical forms ("VORA-X200")."""
        def fix(m: re.Match) -> str:
            words = m.group(0).split(" ")
            i, out = 0, []
            while i < len(words):
                best = (0.0, None, 1)
                for n in range(min(4, len(words) - i), 0, -1):
                    c = _compact(" ".join(words[i:i + n]))
                    if len(c) < 3:
                        continue
                    for canon, v in self._variants:
                        r = fuzz.ratio(c, v)
                        if r >= 85 and r > best[0]:
                            best = (r, canon, n)
                if best[1]:
                    out.append(best[1])
                    i += best[2]
                else:
                    out.append(words[i])
                    i += 1
            return " ".join(out)
        return _LATIN_RUN.sub(fix, text)

    def expand_query(self, text: str) -> str:
        """Append domain wording for colloquial terms ("temperature" -> "operating conditions degrees Celsius"):
        the KB says "degrees Celsius", users say "temperature". Query untouched when no synonym matches."""
        low, extra = text.lower(), []
        for key, add in self.synonyms.items():
            hit = key in low if re.search(r"[\u4e00-\u9fff]", key) else re.search(rf"\b{re.escape(key)}\b", low)
            if hit and add not in extra:
                extra.append(add)
        return text if not extra else f"{text} {' '.join(extra)}"

    def _dense(self, q: str) -> np.ndarray:
        v = np.array(list(self.emb.embed([q])), dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        scores, ids = self.index.search(v, len(self.chunks))
        d = np.zeros(len(self.chunks), dtype=np.float32)
        d[ids[0]] = scores[0]
        return d

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or self.s.top_k
        q = self.expand_query(self.normalize_query(query))
        dense = np.clip(self._dense(q), 0, 1)
        bm = self.bm25.get_scores(tokenize(q))
        bm = bm / bm.max() if bm.max() > 0 else bm
        score = 0.7 * dense + 0.3 * bm
        order = np.argsort(-score)[:k]
        if score[order[0]] < self.min_score:
            return []
        return [Hit(self.chunks[i]["id"], self.chunks[i]["text"], float(score[i])) for i in order]

    def best_score(self, query: str) -> float:
        q = self.expand_query(self.normalize_query(query))
        bm = self.bm25.get_scores(tokenize(q))
        bm = bm / bm.max() if bm.max() > 0 else bm
        return float((0.7 * np.clip(self._dense(q), 0, 1) + 0.3 * bm).max())
