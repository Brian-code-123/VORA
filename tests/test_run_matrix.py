"""The matrix must not overwrite the per-suite results that hold the real-voice latency (G1r reads those files)."""
from scripts import run_matrix


def test_matrix_cells_write_to_own_dir(monkeypatch, tmp_path):
    seen = []

    def fake_run(suite, aug, split, final, n, kb, faith=False, out_dir=None):
        seen.append(out_dir)
        return {"n": 1, "asr": {"metric": "wer", "value": 0.0}}
    monkeypatch.setattr(run_matrix.eval_suite, "run", fake_run)
    monkeypatch.setattr(run_matrix, "cells", lambda: [("librispeech_clean", "clean", "none")])
    monkeypatch.setattr("sys.argv", ["run_matrix", "--out", str(tmp_path / "m.json")])
    run_matrix.main()
    assert seen == [run_matrix.ROOT / "results" / "suites" / "matrix"]
