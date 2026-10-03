import asyncio

import pytest

from vora.rag import store

from vora.config import Settings

S = Settings()
pytestmark = [pytest.mark.perf, pytest.mark.skipif(not store.exists(S.index_dir), reason="needs models + index")]


def test_two_sessions_p50_le_2x_single_and_memory_bounded():
    from scripts.bench_concurrent import run
    from vora.server import Models
    res = asyncio.run(run(Models.load(S), S, 6))
    print(res)
    assert res["n_concurrent"] >= 6 and res["ratio"] <= 2.0, res
    assert res["uss_delta_mb"] <= 150, res
