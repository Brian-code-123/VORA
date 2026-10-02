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
