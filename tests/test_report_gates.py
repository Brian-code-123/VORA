import json

from scripts.report_gates import evaluate


def write(d, name, obj):
    (d / name).write_text(json.dumps(obj))


def base(d, p50_total=1000, tts_p95=150, mem=400, quiet_load=1.0):
    rows = [{"lang": "en", "total": p50_total} for _ in range(10)]
    write(d, "bench.json", {"host": {"cores": 8, "loadavg_1m": quiet_load}, "rows": rows,
                            "latency_ms_oracle_text": {"tts_first_chunk": {"p50": 100, "p95": tts_p95}}})
    write(d, "memory.json", {"asr_rag_tts": mem, "llm": 700})
    write(d, "asr.json", {"en_librispeech_clean/clean": {"value": 0.09}, "zh_fleurs/clean": {"value": 0.25}})
    write(d, "rag.json", {"top3_dev": {"acc": 0.93}, "top3_heldout": {"acc": 0.86}, "faithfulness": {"acc": 0.9}})


def by_id(gs):
    return {g.id: g for g in gs}


def test_gate_fail_and_unverified(tmp_path):
    base(tmp_path, p50_total=1600)
    g = by_id(evaluate(tmp_path))
    assert g["G1"].ok is False          # p50 1.6 s > 1.5 s
    assert g["G7"].ok is None           # no docker result -> unverified
    assert g["G6"].ok is None           # no concurrent result
    assert g["G5"].ok is False          # faithfulness 0.9 < 0.95
    assert g["G4"].ok is None           # zh AISHELL result missing -> unverified, not silently pass


def test_gate_pass_when_met(tmp_path):
    base(tmp_path)
    g = by_id(evaluate(tmp_path))
    assert g["G1"].ok is True and g["G2"].ok is True and g["G3"].ok is True and g["G1"].quiet is True


def test_quiet_flag_false_when_host_loaded(tmp_path):
    base(tmp_path, quiet_load=9.0)
    assert by_id(evaluate(tmp_path))["G1"].quiet is False


def test_report_gates_reads_current_results():
    from pathlib import Path
    gs = evaluate(Path(__file__).resolve().parent.parent / "results")
    assert {g.id for g in gs} == {f"G{i}" for i in range(1, 9)} | {"G1r"}


def test_latency_gates_are_unverified_not_pass_or_fail_on_a_busy_host(tmp_path):
    base(tmp_path, p50_total=1000, quiet_load=9.0)          # numbers would pass, but the host was busy
    g = by_id(evaluate(tmp_path))
    assert g["G1"].ok is None and "busy host" in g["G1"].measured
    base(tmp_path, p50_total=1700, quiet_load=9.0)          # numbers would fail, but the host was busy
    assert by_id(evaluate(tmp_path))["G1"].ok is None


def test_report_gates_g7_partial_states(tmp_path):
    base(tmp_path)
    ok = {"ok": True}
    write(tmp_path, "deploy.json", {"docker_arm64": ok, "docker_amd64": ok, "limited_core_run": ok})       # no Pi-class run
    g = by_id(evaluate(tmp_path))["G7"]
    assert g.ok is None and "pi_class_measured: unverified" in g.measured and "Jetson: unverified" in g.measured
    write(tmp_path, "deploy.json", {"docker_arm64": ok, "docker_amd64": {"ok": False}, "limited_core_run": ok, "pi_class_measured": ok})
    assert by_id(evaluate(tmp_path))["G7"].ok is False
    write(tmp_path, "deploy.json", {k: ok for k in ("docker_arm64", "docker_amd64", "limited_core_run", "pi_class_measured")})
    g = by_id(evaluate(tmp_path))["G7"]
    assert g.ok is True and "Jetson: unverified" in g.measured     # Jetson is never claimed


def test_report_gates_reads_suite_results(tmp_path):
    base(tmp_path)
    (tmp_path / "suites").mkdir()
    lat = lambda p50, p90, quiet=True: {"quiet": quiet, "latency": {"answered": {"p50": p50, "p90": p90, "n": 14}, "n_refused": 2, "n_answered": 12}}
    write(tmp_path / "suites", "minds14_en_us__clean__test.json", lat(1200, 1600))
    write(tmp_path / "suites", "minds14_zh__clean__test.json", lat(1300, 1700))
    g = by_id(evaluate(tmp_path))["G1r"]
    assert g.ok is True and "en-US 1200/1600" in g.measured and "zh-CN 1300/1700" in g.measured
    write(tmp_path / "suites", "minds14_zh__clean__test.json", lat(1300, 1900))
    assert by_id(evaluate(tmp_path))["G1r"].ok is False
    write(tmp_path / "suites", "minds14_zh__clean__test.json", lat(1300, 1700, quiet=False))
    assert by_id(evaluate(tmp_path))["G1r"].ok is None                    # busy host: unverified
    (tmp_path / "suites" / "minds14_zh__clean__test.json").unlink()
    (tmp_path / "suites" / "minds14_en_us__clean__test.json").unlink()
    assert by_id(evaluate(tmp_path))["G1r"].ok is None                    # no real-voice run yet


def test_g5_uses_the_frozen_blind_set_when_present(tmp_path):
    base(tmp_path)
    write(tmp_path, "rag.json", {"top3_dev": {"acc": 1.0}, "top3_heldout": {"acc": 1.0}, "faithfulness": {"acc": 1.0},
                                 "blind": {"top3": 0.96, "faith_with_refusals": 0.817, "n": 60}, "faithfulness_all": {"acc": 0.89, "n": 100}})
    g = by_id(evaluate(tmp_path))["G5"]
    assert g.ok is False and "89%" in g.measured and "96%" in g.measured and "blind" in g.measured
