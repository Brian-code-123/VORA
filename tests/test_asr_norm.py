import pytest

from scripts.eval_asr import norm

pytest.importorskip("num2words")


def test_num_normaliser_table():
    assert norm("lag behind by 25 to 30 years", "en", numbers=True) == "lag behind by twenty five to thirty years"
    assert norm("born in 1990", "en", numbers=True) == "born in nineteen ninety"
    assert norm("in 2005 it rained", "en", numbers=True) == "in two thousand and five it rained"
    assert norm("3.5 meters", "en", numbers=True) == "three point five meters"
    assert norm("lag behind by 25", "en") == "lag behind by 25"                 # raw scoring unchanged by default
    assert norm("Twenty-Five", "en", numbers=True) == "twenty five"
