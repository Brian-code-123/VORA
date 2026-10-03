import json
import re
from pathlib import Path

import pytest

from vora.rag import store

ROOT = Path(__file__).resolve().parent.parent
BANK = ROOT / "eval_kb" / "bank"
INTENT_NAMES = ["abroad", "address", "app_error", "atm_limit", "balance", "business_loan", "card_issues", "cash_deposit",
                "direct_debit", "freeze", "high_value_payment", "joint_account", "latest_transactions", "pay_bill"]   # MInDS-14 label list


def chunks() -> dict[str, str]:
    return {c["id"]: c["text"] for c in store.parse_kb(BANK)}


def intents() -> dict:
    return json.loads((BANK / "intents.json").read_text(encoding="utf-8"))


def test_bank_kb_covers_all_14_intents_both_langs():
    c, i = chunks(), intents()
    assert list(i) == INTENT_NAMES
    for name, v in i.items():
        en, zh = v["chunk_ids"]
        assert en.startswith("B") and zh.startswith("C") and en[1:] == zh[1:], name   # twin ids differ only by the language letter
        assert en in c and zh in c, name
        assert re.search(r"[一-鿿]", c[zh]) and not re.search(r"[一-鿿]", c[en]), name


def test_intents_ids_exist_and_kw_in_chunk():
    c = chunks()
    for name, v in intents().items():
        for lang, cid in zip(("en", "zh"), v["chunk_ids"]):
            assert v["kw"][lang], (name, lang)
            for w in v["kw"][lang]:
                assert w.lower() in c[cid].lower(), (name, lang, w)


def test_bank_chunks_are_distinct_and_not_the_demo_product():
    c = chunks()
    assert len(c) == 28 and len(set(c.values())) == 28
    assert not any("VORA Box" in t or "X200" in t for t in c.values())


def test_ingest_ids_match_parser_regex():
    # store._LINE only accepts one letter + digits: B01 / C01 (a two-letter prefix would be silently dropped)
    lines = [l for f in BANK.glob("*.md") for l in f.read_text(encoding="utf-8").splitlines() if l.startswith("[")]
    assert len(lines) == 28 and len(chunks()) == 28


@pytest.mark.skipif(not store.exists(ROOT / "index_bank"), reason="python scripts/build_bank_index.py not run")
def test_retriever_domain_isolated():
    from vora.config import Settings
    from vora.rag.retriever import Retriever
    r = Retriever(Settings(kb_dir=BANK, index_dir=ROOT / "index_bank"))
    for q in ("how do I freeze my card", "我想查询余额", "what is the warranty on the x200"):
        assert all(h.chunk_id[0] in "BC" for h in r.search(q))
    assert [h.chunk_id for h in r.search("how do I freeze my card")][0] == "B10"
