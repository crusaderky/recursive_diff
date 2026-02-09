from __future__ import annotations

import gc
import threading

import psutil


class MemoryMonitor:
    """Measures peak RAM usage of the current process.

    Spawns a background thread that samples memory usage every 0.01 seconds.

    Usage::

        with MemoryMonitor() as mm:
            # ... do work ...
        # Assert that peak memory usage increased by less than 10 MiB
        mm.assert_peak(10 * 2**20)
    """

    start: int = 0
    peak: int = 0

    def __enter__(self) -> MemoryMonitor:
        self._start_event = threading.Event()
        self._stop_event = threading.Event()
        self._proc = psutil.Process()
        self._thread = threading.Thread(target=self._monitor)
        self._thread.start()
        self._start_event.wait()
        return self

    def _monitor(self) -> None:
        gc.collect()
        self.start = self.peak = self._proc.memory_info().rss
        self._start_event.set()
        while not self._stop_event.is_set():
            spot = self._proc.memory_info().rss
            self.peak = max(self.peak, spot)
            self._stop_event.wait(0.01)
            gc.collect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        self._stop_event.set()
        self._thread.join()

    @property
    def delta(self) -> int:
        """Increase in memory usage from start to peak, in bytes."""
        return self.peak - self.start

    def assert_peak(self, max_delta: int) -> None:
        """Assert that the increase in memory usage from start to peak is less than
        max_delta bytes."""
        assert self.delta < max_delta, str(self)

    def __repr__(self) -> str:
        def fmt(b: int) -> str:
            return f"{b / 2**20:.1f} MiB"

        return (
            f"MemoryMonitor(start={fmt(self.start)}, "
            f"peak={fmt(self.peak)}, delta={fmt(self.delta)})"
        )
