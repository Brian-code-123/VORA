"""Is this host quiet enough for latency/memory measurements, and a stable memory metric (USS)."""
import os

import psutil


def is_quiet(load1: float | None = None, cores: int | None = None, max_ratio: float = 0.5) -> bool:
    load1 = os.getloadavg()[0] if load1 is None else load1
    cores = cores or psutil.cpu_count(logical=True) or 1
    return load1 <= max_ratio * cores


def perf_skip_reason(load1: float | None = None, cores: int | None = None) -> str | None:
    """None = run. VORA_FORCE_PERF=1 overrides (results then carry quiet=false)."""
    if os.environ.get("VORA_FORCE_PERF") == "1" or is_quiet(load1, cores):
        return None
    load = os.getloadavg()[0] if load1 is None else load1
    return f"busy host (load {load:.1f}): perf numbers would be noise; set VORA_FORCE_PERF=1 to force"


def uss_mb(pid: int | None = None) -> float:
    """Unique set size (what freeing the process returns). RSS fluctuates with macOS memory compression."""
    p = psutil.Process(pid)
    try:
        return p.memory_full_info().uss / 2**20
    except (psutil.AccessDenied, AttributeError):
        return p.memory_info().rss / 2**20   # flagged by callers via results[...]["metric"]
