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
