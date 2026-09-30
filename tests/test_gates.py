from scripts.check_gates import evaluate

MB = 1024 * 1024


def test_gate_table_flags_oversize():
    res = evaluate({"asr_en": 60 * MB, "tts_en": 10 * MB})
    assert res["asr_en"]["ok"] is False
    assert res["tts_en"]["ok"] is True


def test_unknown_key_rejected():
    import pytest
    with pytest.raises(KeyError):
        evaluate({"nope": 1})
