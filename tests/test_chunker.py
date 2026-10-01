from vora.chunker import SentenceChunker


def run(tokens):
    c = SentenceChunker()
    out = []
    for t in tokens:
        out += c.push(t)
    return out, c.flush()


def test_first_clause_emitted_early():
    out, tail = run(list("Absolutely sure, the warranty lasts two years. Anything else?"))
    assert out[0] == "Absolutely sure,"  # emitted at the clause, before sentence end
    assert out[1] == "the warranty lasts two years."
    assert out[2] == "Anything else?" and tail == []


def test_first_clause_cjk_threshold():
    out, _ = run(list("好的，保養期係兩年。"))
    assert out == ["好的，保養期係兩年。"] or out[0].endswith("，")
    out2, _ = run(list("保養期限係兩年，請保留收據。"))
    assert out2[0] == "保養期限係兩年，"


def test_decimal_not_split():
    out, tail = run(list("版本3.5很好。"))
    assert out + tail == ["版本3.5很好。"]


def test_flush_returns_tail():
    out, tail = run(["Hello there"])
    assert out == [] and tail == ["Hello there"]


def test_long_unpunctuated_run_force_flushed():
    out, tail = run(["字"] * 130)
    assert [len(x) for x in out] == [60, 60] and tail == ["字" * 10]


def test_empty_and_whitespace_tokens():
    out, tail = run(["", " ", "\n", "Hi", " ", ""])
    assert out == [] and tail == ["Hi"]


def test_abbreviation_not_split():
    out, tail = run(list("Use e.g. the app. Done."))
    assert out + tail == ["Use e.g. the app.", "Done."]


def test_first_chunk_split_after_four_words_when_no_punctuation():
    out, tail = run(list("The warranty on the VORA-X200 is two years."))
    assert out[0] == "The warranty on the"
    assert " ".join(out + tail) == "The warranty on the VORA-X200 is two years."


def test_short_first_words_not_split():
    out, tail = run(list("It is on a red day."))
    assert out + tail == ["It is on a red day."]


def test_first_chunk_words_is_configurable():
    c = SentenceChunker(first_words=2)
    out = []
    for t in "The warranty on the VORA-X200 is two years.":
        out += c.push(t)
    assert out[0] == "The warranty"
    assert " ".join(out + c.flush()) == "The warranty on the VORA-X200 is two years."


def test_first_words_two_does_not_split_short_answers():
    out, tail = [], []
    c = SentenceChunker(first_words=2)
    for t in "Yes it is.":
        out += c.push(t)
    assert out + c.flush() == ["Yes it is."]


def test_short_first_words_still_split_when_seven_chars():
    for text, first in (("You can import documents.", "You can"), ("The box draws five watts.", "The box"),
                        ("It does not support Cantonese.", "It does")):
        c = SentenceChunker(first_words=2)
        out = []
        for t in text:
            out += c.push(t)
        assert out[0] == first, (text, out)


def test_progressive_chunks_one_then_three_words():
    c = SentenceChunker(first_words=1, second_words=3)
    out = []
    for t in "The warranty on the VORA-X200 is two years. Anything else?":
        out += c.push(t)
    assert out[:2] == ["The", "warranty on the"]
    assert " ".join(out + c.flush()) == "The warranty on the VORA-X200 is two years. Anything else?"


def test_progressive_does_not_split_after_sentence_end_or_short_tail():
    c = SentenceChunker(first_words=1, second_words=3)
    out = []
    for t in "The warranty is long. Yes it is.":
        out += c.push(t)
    tail = c.flush()
    assert " ".join(out + tail) == "The warranty is long. Yes it is."
    assert out[0] == "The"
