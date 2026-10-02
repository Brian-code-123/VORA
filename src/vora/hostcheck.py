"""Is this host quiet enough for latency/memory measurements, and a stable memory metric (USS)."""
import os

import psutil


MAX_BUSY_CPU_PCT = 30.0


def is_quiet(load1: float | None = None, cores: int | None = None, max_ratio: float = 0.5) -> bool:
    """With an explicit load1 (tests, recorded results): load average vs cores. Without: a 1 s CPU sample, because the
    1-minute load average lags and includes our own previous runs."""
    if load1 is None:
        return psutil.cpu_percent(interval=1.0) <= MAX_BUSY_CPU_PCT
    cores = cores or psutil.cpu_count(logical=True) or 1
    return load1 <= max_ratio * cores


def perf_skip_reason(load1: float | None = None, cores: int | None = None) -> str | None:
    """None = run. VORA_FORCE_PERF=1 overrides (results then carry quiet=false)."""
    if os.environ.get("VORA_FORCE_PERF") == "1" or is_quiet(load1, cores):
        return None
    what = f"load {load1:.1f}" if load1 is not None else f"CPU >{MAX_BUSY_CPU_PCT:.0f}% busy"
    return f"busy host ({what}): perf numbers would be noise; set VORA_FORCE_PERF=1 to force"


def uss_mb(pid: int | None = None) -> float:
    """Unique set size (what freeing the process returns). RSS fluctuates with macOS memory compression."""
    p = psutil.Process(pid)
    try:
        return p.memory_full_info().uss / 2**20
    except (psutil.AccessDenied, AttributeError):
        return p.memory_info().rss / 2**20   # flagged by callers via results[...]["metric"]


def write_result(path, obj) -> bool:
    """Perf/eval tests only overwrite tracked results/*.json when VORA_WRITE_RESULTS=1 (a normal test run must not
    silently change the numbers the report and the gates are built from)."""
    import json
    from pathlib import Path
    if os.environ.get("VORA_WRITE_RESULTS") != "1":
        return False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj))
    return True


def _sample() -> tuple[float, float]:
    """(system CPU %, this process tree's CPU % normalised to the whole machine)."""
    me = psutil.Process()
    procs = [me] + me.children(recursive=True)
    own = sum(p.cpu_percent(interval=None) for p in procs) / (psutil.cpu_count(logical=True) or 1)
    return psutil.cpu_percent(interval=None), own


class QuietMonitor:
    """Samples CPU every `interval` s DURING a measurement and keeps the worst load caused by OTHER processes
    (system minus ourselves). is_quiet() before/after a run says nothing about what ran in between."""

    def __init__(self, interval: float = 1.0, sampler=_sample, max_other_pct: float = MAX_BUSY_CPU_PCT):
        import threading
        self._interval, self._sampler, self._max, self._stop = interval, sampler, max_other_pct, threading.Event()
        self._worst = 0.0
        self._samples: list[float] = []
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self._sampler()                                   # prime psutil's counters
        while not self._stop.wait(self._interval):
            sysp, own = self._sampler()
            other = round(max(0.0, sysp - own), 1)
            self._samples.append(other)
            self._worst = max(self._worst, other)

    def start(self) -> None:
        self._t.start()

    def stop(self) -> dict:
        self._stop.set()
        self._t.join(timeout=5)
        s = sorted(self._samples)
        p90 = s[min(len(s) - 1, int(0.9 * len(s)))] if s else 0.0
        # quiet = the 90th percentile of other processes' CPU is low: a one-second burst from a mail client must not
        # void a 10-minute run, but sustained load must (the max is reported next to it)
        return {"quiet": p90 <= self._max, "max_other_cpu_pct": self._worst, "p90_other_cpu_pct": p90}
