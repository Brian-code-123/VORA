import numpy as np
import pytest

from vora.config import Settings
from vora.rag import ingest, store
from vora.rag.retriever import Retriever
from vora.rag.store import Hit, split_passages

S = Settings()


def test_split_passages_titles_and_sentences():
    p = split_passages("Warranty: the X200 warranty is 2 years. The M100 warranty is 1 year. Water damage is not covered.")
    assert [s for s, _ in p] == ["Warranty: the X200 warranty is 2 years.", "The M100 warranty is 1 year.", "Water damage is not covered."]
    assert p[0][1] == p[0][0]                                   # the first sentence already carries the title
    assert p[1][1] == "Warranty: The M100 warranty is 1 year."  # later sentences get the title so they stand alone
    zh = split_passages("保修：X200 保修 2 年。M100 保修 1 年。")
    assert zh[1] == ("M100 保修 1 年。", "保修：M100 保修 1 年。")
    assert split_passages("No title here. Second sentence.")[1][1] == "Second sentence."   # no "title:" -> unchanged
    assert split_passages("") == []


def test_hit_has_focus_default():
    assert Hit("E01", "text", 0.5).focus == ""


def test_index_roundtrip_has_passages_per_language(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.md").write_text("[E01] Power: the box uses 12 V. It draws 5 W idle.\n[Z01] 电源：盒子使用 12 V。待机约 5 瓦。\n", encoding="utf-8")
    ingest.ingest(kb, tmp_path / "idx")
    lanes, chunks, meta = store.load(tmp_path / "idx")
    assert set(lanes) == {"en", "zh"}
    for lang in ("en", "zh"):
        assert len(lanes[lang].psent) == 2 and lanes[lang].pvec.shape[0] == 2 and list(lanes[lang].pchunk) == [lanes[lang].pos[0]] * 2


@pytest.mark.skipif(not store.exists(S.index_dir), reason="run: python -m vora.rag.ingest")
class TestWithIndex:
    @pytest.fixture(scope="class")
    def r(self):
        return Retriever(S)

    def test_focus_is_the_best_matching_sentence(self, r):
        h = r.search("how loud is the m100 speaker")[0]
        assert h.chunk_id == "E17" and "5 W speaker" in h.focus
        h = r.search("x200 拾音距离")[0]
        assert h.chunk_id == "Z16" and "5 米" in h.focus

    def test_passage_scoring_ranks_the_answer_chunk_first(self, r):
        # sentence-level scores: the sentence "A hardware mute switch on top of the box ..." beats the LED chunk
        assert r.search("where is the hardware mute switch")[0].chunk_id == "E06"
        assert "E17" in [h.chunk_id for h in r.search("can I change how loud it talks by voice")]
