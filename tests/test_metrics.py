from vora.metrics import LatencyTrace


def test_trace_report_orders_marks():
    t = iter([0.0, 0.5, 0.9, 1.3])
    tr = LatencyTrace(clock=lambda: next(t))
    tr.mark("speech_end")      # 0.0
    tr.mark("final")           # 0.5
    tr.mark("first_token")     # 0.9
    tr.mark("first_audio")     # 1.3
    r = tr.report()
    assert r["asr_final"] == 500.0
    assert r["rag_first_token"] == 400.0
    assert r["tts_first_chunk"] == 400.0
    assert r["total"] == 1300.0


def test_report_with_missing_marks_omits_keys():
    tr = LatencyTrace(clock=lambda: 1.0)
    tr.mark("speech_end")
    assert tr.report() == {}
