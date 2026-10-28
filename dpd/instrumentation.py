"""High-resolution performance instrumentation and telemetry for DPD."""

from __future__ import annotations

import time
from collections.abc import Callable


class HighResolutionProfiler:
    """Measures execution duration using high-resolution performance counters."""

    @staticmethod
    def now_ns() -> int:
        """Returns the current hardware performance counter in nanoseconds."""
        return time.perf_counter_ns()

    @staticmethod
    def now_sec() -> float:
        """Returns the current hardware performance counter in fractional seconds."""
        return time.perf_counter()

    @staticmethod
    def benchmark_callable(
        fn: Callable[[], None],
        warmup_runs: int = 5,
        benchmark_runs: int = 20,
    ) -> tuple[float, float, list[float]]:
        """Executes a benchmark with mandatory warmup runs and high-resolution timing.
        
        Args:
            fn: The zero-argument callable to benchmark.
            warmup_runs: Number of discarded warmup runs (minimum 5).
            benchmark_runs: Number of timed evaluation runs.
            
        Returns:
            Tuple of (mean_time_sec, std_dev_sec, all_durations_sec).
        """
        warmup_runs = max(warmup_runs, 5)

        # Discard warmup runs
        for _ in range(warmup_runs):
            fn()

        durations: list[float] = []
        for _ in range(benchmark_runs):
            t0 = time.perf_counter_ns()
            fn()
            t1 = time.perf_counter_ns()
            durations.append((t1 - t0) * 1e-9)

        mean_dur = sum(durations) / len(durations)
        variance = sum((d - mean_dur) ** 2 for d in durations) / max(1, len(durations) - 1)
        std_dur = variance ** 0.5

        return mean_dur, std_dur, durations
