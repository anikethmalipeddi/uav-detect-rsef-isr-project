from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class LatencySummary:
    mean_ms: float
    median_ms: float
    p95_ms: float
    n: int


def _summary(ms: list[float]) -> LatencySummary:
    arr = np.array(ms, dtype=np.float64)
    return LatencySummary(
        mean_ms=float(arr.mean()),
        median_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        n=int(arr.size),
    )


def measure_wall_ms(fn: Callable[[], None], warmup: int = 10, runs: int = 200) -> LatencySummary:
    for _ in range(warmup):
        fn()
    ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        ms.append((t1 - t0) * 1000.0)
    return _summary(ms)


def measure_cuda_ms(fn: Callable[[], None], warmup: int = 10, runs: int = 200) -> LatencySummary:
    if torch is None or not torch.cuda.is_available():
        return measure_wall_ms(fn, warmup=warmup, runs=runs)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    ms = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        ms.append(float(start.elapsed_time(end)))
    return _summary(ms)
