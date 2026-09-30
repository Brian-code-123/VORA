import math

from scripts.bench import percentile, rtf


def test_rtf_and_percentile():
    assert rtf(0.5, 2.0) == 0.25
    xs = [10, 20, 30, 40, 100]
    assert percentile(xs, 50) == 30
    assert math.isclose(percentile(xs, 95), 88.0)
    assert math.isnan(percentile([], 50))
