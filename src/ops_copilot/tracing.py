"""Langfuse tracing that degrades to a no-op when no keys are configured.

Every run is one trace (session = LangGraph thread id, user = requester);
every model call is a `generation` with token usage and USD cost, every tool
call and retrieval is a span. That is what makes cost-per-run and "what did
the agent actually do" answerable after the fact.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any


class _NullObservation:
    def update(self, **kwargs: Any) -> None:
        pass


class Tracer:
    def __init__(self) -> None:
        self._lf = None
        if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
            from langfuse import get_client

            self._lf = get_client()

    @property
    def enabled(self) -> bool:
        return self._lf is not None

    @contextmanager
    def run(self, name: str, *, user_id: str, session_id: str, input: Any) -> Iterator[Any]:
        if not self._lf:
            yield _NullObservation()
            return
        from langfuse import propagate_attributes

        with propagate_attributes(user_id=user_id, session_id=session_id, trace_name=name):
            with self._lf.start_as_current_observation(name=name, as_type="agent", input=input) as obs:
                yield obs

    def observation(self, name: str, as_type: str = "span", **kwargs: Any):
        if not self._lf:
            return nullcontext(_NullObservation())
        return self._lf.start_as_current_observation(name=name, as_type=as_type, **kwargs)

    def trace_url(self) -> str | None:
        if not self._lf:
            return None
        try:
            return self._lf.get_trace_url()
        except Exception:  # noqa: BLE001 - a missing URL must never fail a run
            return None

    def flush(self) -> None:
        if self._lf:
            self._lf.flush()


_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer
