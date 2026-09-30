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
