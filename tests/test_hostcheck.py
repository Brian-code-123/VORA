from vora.hostcheck import is_quiet, perf_skip_reason, uss_mb


def test_is_quiet_false_when_load_high():
    assert is_quiet(load1=6.0, cores=8) is False


def test_is_quiet_true_when_idle():
    assert is_quiet(load1=1.0, cores=8) is True


def test_perf_skip_reason_on_busy_host_and_override(monkeypatch):
    assert "busy" in perf_skip_reason(load1=9.0, cores=8)
    assert perf_skip_reason(load1=0.5, cores=8) is None
    monkeypatch.setenv("VORA_FORCE_PERF", "1")
    assert perf_skip_reason(load1=9.0, cores=8) is None


def test_uss_mb_positive_for_self():
    assert uss_mb() > 5


def test_is_quiet_uses_instant_cpu_when_load_not_given(monkeypatch):
    """The 1-minute load average lags and counts our own earlier runs; a 1 s CPU sample says what is busy NOW."""
    import vora.hostcheck as h
    monkeypatch.setattr(h.psutil, "cpu_percent", lambda interval=None: 12.0)
    assert h.is_quiet() is True
    monkeypatch.setattr(h.psutil, "cpu_percent", lambda interval=None: 65.0)
    assert h.is_quiet() is False


def test_quiet_monitor_ignores_our_own_cpu_and_flags_other_load(monkeypatch):
    import vora.hostcheck as h
    seq = iter([(90.0, 85.0)] + [(95.0, 40.0)] * 100)    # (system %, own %): first sample is all us, then 55% from others, sustained

    m = h.QuietMonitor(interval=0.005, sampler=lambda: next(seq, (95.0, 40.0)))
    m.start()
    import time
    time.sleep(0.3)
    r = m.stop()
    assert r["max_other_cpu_pct"] == 55.0 and r["quiet"] is False


def test_quiet_monitor_one_short_burst_does_not_disqualify_a_run():
    import time
    import vora.hostcheck as h
    seq = iter([(10.0, 5.0)] + [(8.0, 5.0)] * 40 + [(90.0, 10.0)] + [(8.0, 5.0)] * 40)       # one 80% burst of another app
    m = h.QuietMonitor(interval=0.002, sampler=lambda: next(seq, (8.0, 5.0)))
    m.start()
    time.sleep(0.4)
    r = m.stop()
    assert r["max_other_cpu_pct"] >= 80 and r["p90_other_cpu_pct"] <= 5 and r["quiet"] is True


def test_quiet_monitor_quiet_when_only_we_run():
    import time
    import vora.hostcheck as h
    m = h.QuietMonitor(interval=0.01, sampler=lambda: (80.0, 78.0))
    m.start()
    time.sleep(0.1)
    assert m.stop()["quiet"] is True


def test_write_result_only_when_enabled(tmp_path, monkeypatch):
    import vora.hostcheck as h
    monkeypatch.delenv("VORA_WRITE_RESULTS", raising=False)
    assert h.write_result(tmp_path / "x.json", {"a": 1}) is False and not (tmp_path / "x.json").exists()
    monkeypatch.setenv("VORA_WRITE_RESULTS", "1")
    assert h.write_result(tmp_path / "x.json", {"a": 1}) is True and (tmp_path / "x.json").exists()
