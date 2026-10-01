import pytest

from vora.hostcheck import perf_skip_reason


def pytest_runtest_setup(item):
    if item.get_closest_marker("perf"):
        reason = perf_skip_reason()
        if reason:
            pytest.skip(reason)
