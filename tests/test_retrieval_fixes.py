import hashlib
import json

import pytest

from vora.rag import store

from vora.config import ROOT, Settings
from vora.rag.retriever import Retriever

S = Settings()
pytestmark = pytest.mark.skipif(not store.exists(S.index_dir), reason="run: python -m vora.rag.ingest")


@pytest.fixture(scope="module")
def r():
    return Retriever(S)


def ids(hits):
    return [h.chunk_id for h in hits]


def test_expand_query_latin_key_adjacent_to_cjk(r):
    # \b treats CJK as word characters, so "的wifi" had no boundary and the Wi-Fi synonym never fired (zh miss in G5)
    for q in ("m100能连5g的wifi吗", "用wifi", "wifi吗", "怎么连wifi"):
        assert "2.4 GHz" in r.expand_query(q), q
    assert r.expand_query("awifi") == "awifi"          # still a real word boundary on the Latin side
    assert r.expand_query("wifi6") == "wifi6"
    assert "2.4 GHz" in r.expand_query("does it have wifi?")


def test_normalize_query_boundaries_unchanged_for_cjk_neighbours(r):
    assert "VORA-X200" in r.normalize_query("vora x 200保修期")
    assert "VORA-M100" in r.normalize_query("m100能连5g吗")


def test_three_known_failures_now_pass(r):
    assert {"E07", "Z07"} & set(ids(r.search("m100能连5g的wifi吗")))        # was: no hits at all (best 0.603 < 0.67)
    assert {"E16", "Z16"} & set(ids(r.search("how far can the m100 hear me")))   # was: E12, E02, E03
    assert ids(r.search("x200 拾音范围"))[0] in {"E16", "Z16"}              # was: Z18 above Z16


def test_synonyms_lint_general_only():
    syn = json.loads((S.kb_dir / "synonyms.json").read_text(encoding="utf-8"))
    questions = set()
    for f in ("rag_qa.jsonl", "faithfulness.jsonl", "rag_blind.jsonl"):
        questions |= {json.loads(l)["q"].lower().strip() for l in (ROOT / "eval" / f).read_text(encoding="utf-8").splitlines() if l.strip()}
    for key, add in syn.items():
        assert len(key.split()) <= 3 and len(key) <= 16, f"synonym key looks like a sentence: {key!r}"
        assert key.lower() not in questions, f"synonym key is an eval question: {key!r}"
        assert add.strip() and key != add


@pytest.mark.parametrize("name,sha", [("rag_blind", "BLIND.sha256"), ("rag_blind2", "BLIND2.sha256"), ("rag_blind3", "BLIND3.sha256"), ("rag_blind4", "BLIND4.sha256")])
def test_blind_set_hash_unchanged(name, sha):
    want = (ROOT / "eval" / sha).read_text().split()[0]
    got = hashlib.sha256((ROOT / "eval" / f"{name}.jsonl").read_bytes()).hexdigest()
    assert got == want, f"eval/{name}.jsonl was edited: frozen sets change only with a ruling in docs/rulings.md"


@pytest.mark.parametrize("name", ["rag_blind", "rag_blind2", "rag_blind3", "rag_blind4"])
def test_blind_set_shape(name):
    rows = [json.loads(l) for l in (ROOT / "eval" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 60 and sum(x["kind"] == "neg" for x in rows) == 12 and sum(x["kind"] == "offtopic" for x in rows) == 10
    assert {x["lang"] for x in rows} == {"en", "zh"} and all(x["kw"] and x["chunk_id"] for x in rows if x["kind"] != "offtopic")


def test_blind3_and_4_do_not_overlap_any_other_eval_set():
    """blind v1 duplicated 7 questions of the older sets (one more reason it is exposed); v2 and v3 must share none."""
    qs = {}
    for f in ("rag_qa", "faithfulness", "rag_blind", "rag_blind2", "rag_blind3", "rag_blind4", "offtopic_dev"):
        qs[f] = [json.loads(l)["q"].lower().strip() for l in (ROOT / "eval" / f"{f}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for new in ("rag_blind2", "rag_blind3", "rag_blind4"):
        for other in ("rag_qa", "faithfulness", "rag_blind", "rag_blind2", "rag_blind3", "rag_blind4", "offtopic_dev"):
            if other != new:
                assert not set(qs[new]) & set(qs[other]), (new, other, set(qs[new]) & set(qs[other]))


@pytest.mark.parametrize("name", ["rag_blind", "rag_blind2"])   # tuning sets only: the held-out sets are reported by eval_rag.py, never asserted
def test_tuning_set_offtopic_is_refused(r, name):
    off = [json.loads(l)["q"] for l in (ROOT / "eval" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if '"offtopic"' in l]
    assert len(off) == 10 and sum(r.search(q) == [] for q in off) / len(off) >= 0.9


def test_min_score_calibrated_per_kb():
    bank = ROOT / "index_bank" / "meta.json"
    if not bank.exists():
        pytest.skip("python scripts/build_bank_index.py not run")
    m = json.loads(bank.read_text())
    assert set(m["min_score"]) == {"en", "zh"} and all(v >= 0.9 for v in m["offtopic_rejected"].values())
