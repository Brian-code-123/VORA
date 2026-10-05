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
from vora.rag.store import LANG_MODELS, Hit, lang_of

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
        self.lanes, self.chunks, meta = store.load(index_dir or settings.index_dir)
        ms = meta.get("min_score", settings.min_score)
        self.min_score = ms if isinstance(ms, dict) else {lang: ms for lang in self.lanes}   # per language
        self._emb: dict[str, TextEmbedding] = {}      # loaded on first query in that language
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
            # Latin keys need a Latin-side boundary. \b is wrong here: CJK counts as \w, so "的wifi" had no boundary
            hit = key in low if re.search(r"[\u4e00-\u9fff]", key) else re.search(rf"(?<![A-Za-z0-9]){re.escape(key)}(?![A-Za-z0-9])", low)
            if hit and add not in extra:
                extra.append(add)
        return text if not extra else f"{text} {' '.join(extra)}"

    def _lane(self, q: str) -> str:
        lang = lang_of(q)
        return lang if lang in self.lanes else next(iter(self.lanes))   # KB without chunks in that language: use what exists

    def _embedder(self, lang: str) -> TextEmbedding:
        if lang not in self._emb:
            try:      # offline box: never pay a network round trip (measured 4 s on the first Chinese question)
                self._emb[lang] = TextEmbedding(LANG_MODELS[lang], threads=1, local_files_only=True)
            except Exception:  # noqa: BLE001 - not cached yet: download once
                self._emb[lang] = TextEmbedding(LANG_MODELS[lang], threads=1)
        return self._emb[lang]

    def _dense(self, q: str, lang: str) -> tuple[np.ndarray, dict[int, str]]:
        """Per-chunk similarity: the chunk itself averaged with its best sentence (Settings.passage_weight), and that sentence."""
        v = np.array(list(self._embedder(lang).embed([q])), dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        lane = self.lanes[lang]
        scores, ids = lane.index.search(v, len(lane.pos))
        d = np.zeros(len(self.chunks), dtype=np.float32)
        d[lane.pos[ids[0]]] = scores[0]
        focus: dict[int, str] = {}
        if len(lane.psent) and self.s.passage_weight > 0:
            ps = lane.pvec @ v[0]
            best = np.full(len(self.chunks), -1.0, dtype=np.float32)
            for k, c in enumerate(lane.pchunk):
                if ps[k] > best[c]:
                    best[c], focus[int(c)] = ps[k], lane.psent[k]
            has = best > -1.0
            d[has] = (1 - self.s.passage_weight) * d[has] + self.s.passage_weight * best[has]
        return d, focus

    def _bm25(self, q: str) -> np.ndarray:
        toks = tokenize(q)
        bm = self.bm25.get_scores(toks)
        if self.s.bm25_norm == "idf":    # fraction of the query's information content that a chunk matches
            den = sum(self.bm25.idf.get(t, 0.0) for t in set(toks)) * (self.bm25.k1 + 1)
            return bm / den if den > 0 else bm * 0
        return bm / bm.max() if bm.max() > 0 else bm

    def _score(self, q: str) -> tuple[np.ndarray, str, dict[int, str]]:
        lang = self._lane(q)
        w = self.s.dense_weight
        dense, focus = self._dense(q, lang)
        return w * np.clip(dense, 0, 1) + (1 - w) * self._bm25(q), lang, focus

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or self.s.top_k
        score, lang, focus = self._score(self.expand_query(self.normalize_query(query)))
        order = np.argsort(-score)[:k]
        if score[order[0]] < self.min_score.get(lang, self.s.min_score):
            return []
        return [Hit(self.chunks[i]["id"], self.chunks[i]["text"], float(score[i]), focus.get(int(i), "")) for i in order]

    def best_score(self, query: str) -> float:
        return float(self._score(self.expand_query(self.normalize_query(query)))[0].max())
