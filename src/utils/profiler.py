"""Lightweight, removable profiling for the RAG pipeline.

This module provides two interchangeable instrumentation helpers that
record wall-clock timings into an *active* :class:`Profiler`:

* ``with measure("stage name"):``  - a context manager, for timing a block
* ``@profile("stage name")``        - a decorator, for timing a whole function

Both are **no-ops when no profiler is active**, so instrumented code stays
correct and zero-cost in production / tests unless a profiler is explicitly
activated via :func:`use_profiler`. That makes the instrumentation trivial
to remove later: delete the ``with measure(...)`` / ``@profile`` lines and
the single ``use_profiler`` block - nothing else in the pipeline changes.

The active profiler is stored in a :class:`contextvars.ContextVar`, so each
async task / thread of execution sees its own profiler (safe under FastAPI).

All timings use :func:`time.perf_counter` (monotonic, highest-resolution)
- never :func:`time.time`, which is wall-clock and can go backwards.
"""

from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Iterator, Optional

# Per-context active profiler. None when profiling is disabled.
_active: contextvars.ContextVar[Optional["Profiler"]] = contextvars.ContextVar(
    "rag_active_profiler", default=None
)


class Profiler:
    """Accumulates named stage timings and renders a summary table.

    A single instance is meant to scope one end-to-end operation (e.g. one
    ``RAGPipeline.answer()`` call). Stages are recorded in insertion order
    so the summary reads in pipeline order; re-entrant / repeated records
    of the same name accumulate (count + total seconds).
    """

    def __init__(self) -> None:
        # name -> {"count": int, "seconds": float}; dict preserves insertion order
        self._stages: dict[str, dict[str, float]] = {}

    def record(self, name: str, seconds: float) -> None:
        """Accumulate ``seconds`` under stage ``name`` (created on first use)."""
        entry = self._stages.get(name)
        if entry is None:
            self._stages[name] = {"count": 1, "seconds": seconds}
        else:
            entry["count"] += 1
            entry["seconds"] += seconds

    @property
    def stages(self) -> dict[str, dict[str, float]]:
        """Recorded stages as ``{name: {count, seconds}}`` (insertion-ordered)."""
        return self._stages

    def reset(self) -> None:
        """Clear all recorded stages."""
        self._stages.clear()

    def summary(self, total_seconds: Optional[float] = None) -> str:
        """Render the timing summary as a printable table.

        Args:
            total_seconds: The measured end-to-end wall-clock time. When
                provided it is printed as the final TOTAL row and used as
                the denominator for the percentage column. When omitted,
                the sum of recorded stages is used as the total.

        Returns:
            A multi-line string containing the table and a bottleneck line.
        """
        recorded_total = sum(s["seconds"] for s in self._stages.values())
        total = total_seconds if total_seconds is not None else recorded_total
        total_ms = total * 1000.0

        # Bottleneck = slowest *component* stage. Skip the end-to-end total
        # (recorded separately by the pipeline) so it isn't reported as its
        # own bottleneck at ~100%.
        _TOTAL_NAMES = {"8. total end-to-end", "total end-to-end", "total"}
        bottleneck_name, bottleneck_ms = "n/a", 0.0
        for name, entry in self._stages.items():
            if name.lower() in {n.lower() for n in _TOTAL_NAMES}:
                continue
            ms = entry["seconds"] * 1000.0
            if ms > bottleneck_ms:
                bottleneck_ms, bottleneck_name = ms, name

        lines: list[str] = []
        lines.append("============== RAG PIPELINE TIMING SUMMARY ==============")
        lines.append(f"{'Stage':<38}| {'ms':>10} | {'seconds':>10} | {'%':>7}")
        lines.append("-" * 72)
        for name, entry in self._stages.items():
            ms = entry["seconds"] * 1000.0
            pct = (entry["seconds"] / total * 100.0) if total > 0 else 0.0
            count = entry["count"]
            label = name if count == 1 else f"{name} (x{count})"
            lines.append(
                f"{label:<38}| {ms:>10.2f} | {entry['seconds']:>10.6f} | {pct:>6.2f}%"
            )
        lines.append("-" * 72)
        pct_total = 100.0 if total > 0 else 0.0
        lines.append(
            f"{'TOTAL (end-to-end)':<38}| {total_ms:>10.2f} | {total:>10.6f} | {pct_total:>6.2f}%"
        )
        lines.append("==========================================================")
        if bottleneck_name != "n/a" and total > 0:
            share = bottleneck_ms / total_ms * 100.0 if total_ms > 0 else 0.0
            lines.append(
                f"Bottleneck: {bottleneck_name} - {bottleneck_ms:.2f} ms "
                f"({share:.1f}% of total)"
            )
        return "\n".join(lines)

    def print_summary(self, total_seconds: Optional[float] = None) -> None:
        """Print :meth:`summary` to stdout."""
        print("\n" + self.summary(total_seconds) + "\n")


# --------------------------------------------------------------------------- #
# Activation
# --------------------------------------------------------------------------- #

@contextmanager
def use_profiler(profiler: Profiler) -> Iterator[Profiler]:
    """Activate ``profiler`` for the duration of the ``with`` block.

    Sets the active profiler in the current context (via a ContextVar, so
    nested ``with measure(...)`` blocks anywhere in the call stack - even in
    other modules - record into this profiler). Restores the previous value
    on exit.
    """
    token = _active.set(profiler)
    try:
        yield profiler
    finally:
        _active.reset(token)


def get_active_profiler() -> Optional[Profiler]:
    """Return the currently active profiler, or ``None`` if profiling is off."""
    return _active.get()


# --------------------------------------------------------------------------- #
# Instrumentation helpers (no-op when no profiler is active)
# --------------------------------------------------------------------------- #

@contextmanager
def measure(name: str) -> Iterator[None]:
    """Context manager that times a block into the active profiler.

    Usage::

        with measure("vector similarity search"):
            results = vector_store.similarity_search(query_embedding, top_k=k)

    When no profiler is active this yields immediately with zero overhead
    beyond the context-manager protocol, so it is safe to leave in place.
    """
    profiler = _active.get()
    if profiler is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        profiler.record(name, time.perf_counter() - start)


def profile(name: str) -> Callable[[Callable], Callable]:
    """Decorator that times a function into the active profiler.

    Usage::

        @profile("query embedding")
        def embed(self, text: str) -> list[float]:
            ...

    No-op when no profiler is active (the wrapped function runs as-is).
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            profiler = _active.get()
            if profiler is None:
                return fn(*args, **kwargs)
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                profiler.record(name, time.perf_counter() - start)

        return wrapper

    return decorator