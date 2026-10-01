import time

import pytest

from vora.config import ROOT, Settings
from vora.rag.ingest import load_qa
from vora.rag.retriever import Retriever

S = Settings()
pytestmark = pytest.mark.skipif(not (S.index_dir / "faiss.index").exists(), reason="run: python -m vora.rag.ingest")
QA = load_qa(ROOT / "eval" / "rag_qa.jsonl")


@pytest.fixture(scope="module")
def r():
    return Retriever(S)


def top3(r, split):
    qs = [q for q in QA if q["chunk_id"] and q["split"] == split]
    ok = sum(bool({h.chunk_id for h in r.search(q["q"])} & set(q["chunk_id"])) for q in qs)
    return ok / len(qs), [q["q"] for q in qs if not ({h.chunk_id for h in r.search(q["q"])} & set(q["chunk_id"]))]


def test_top3_accuracy_ge_80pct(r):
    acc, miss = top3(r, "dev")
    assert acc >= 0.8, (acc, miss)


def test_heldout_top3_reported(r):
    acc, miss = top3(r, "heldout")
    print(f"heldout top-3 = {acc:.2f}, misses: {miss}")
    assert acc >= 0.7


def test_exact_product_term_ranks_first(r):
    assert r.search("VORA-X200 保修期")[0].chunk_id in {"E03", "Z03"}


def test_asr_mangled_term_normalised(r):
    assert "VORA-X200" in r.normalize_query("what is the vora x 200 warranty")
    assert "VORA-X200" in r.normalize_query("vora ex 200 保修期")
    assert r.normalize_query("what is the weather") == "what is the weather"


def test_offtopic_returns_empty(r):
    off = [q["q"] for q in QA if not q["chunk_id"]]
    rejected = sum(r.search(q) == [] for q in off) / len(off)
    assert rejected >= 0.9, rejected


@pytest.mark.perf
def test_search_under_50ms(r):
    r.search("warm up")
    t = []
    for _ in range(20):
        t0 = time.perf_counter()
        r.search("how long is the warranty")
        t.append((time.perf_counter() - t0) * 1000)
    t.sort()
    assert t[int(0.95 * len(t)) - 1] < 50, t  # embed+faiss+bm25 all-in; plan target of 20 ms applies to faiss alone


def test_synonym_expansion_adds_domain_terms(r):
    q = r.expand_query("what temperature can it operate in")
    assert "degrees" in q.lower() and q.lower().startswith("what temperature")
    assert r.expand_query("how long is the warranty") == "how long is the warranty"   # no synonym -> untouched


def test_synonyms_do_not_hurt_retrieval(r):
    acc_dev, miss = top3(r, "dev")
    assert acc_dev >= 0.92, miss
    acc_ho, miss_ho = top3(r, "heldout")
    assert acc_ho >= 0.85, miss_ho
