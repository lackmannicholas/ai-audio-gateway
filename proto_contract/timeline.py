"""A lightweight, nesting-aware timeline of a call.

This is the simplified, dependency-free cousin of the distributed trace a real
deployment would emit (e.g. OpenTelemetry spans, with the gateway starting the
root span and injecting trace context across the gRPC boundary so business-plane
tool calls nest under the exact gateway proxy span that requested them).

Here we just record events with a depth and a monotonic sequence, so you can
print a stack-trace-style view of a turn and *see* the nesting:

    call
      caller_turn
        model_response
          tool_call:consult_thinker        (crosses the wire)
            business.tool:consult_thinker   (executes in business plane)
              business.local:get_menu       (nested, never crosses back)
              business.local:place_order

The point: even this minimal version makes the plane boundary and the nested
fan-out legible. The real one does the same thing with cross-process trace
propagation.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field


@dataclass
class TimelineEvent:
    name: str
    depth: int
    sequence: int
    ts: float
    source: str  # "gateway" | "business"
    detail: dict = field(default_factory=dict)


class Timeline:
    def __init__(self) -> None:
        self._events: list[TimelineEvent] = []
        self._seq = itertools.count()
        self._t0 = time.time()

    def record(self, name: str, depth: int, source: str, **detail) -> None:
        self._events.append(TimelineEvent(
            name=name, depth=depth, sequence=next(self._seq),
            ts=time.time() - self._t0, source=source, detail=detail,
        ))

    def render(self) -> str:
        lines = []
        for e in self._events:
            indent = "  " * e.depth
            tag = "·gw" if e.source == "gateway" else "·bz"
            ms = f"{e.ts * 1000:7.1f}ms"
            lines.append(f"{ms} {indent}{e.name} {tag}")
        return "\n".join(lines)

    @property
    def events(self) -> list[TimelineEvent]:
        return list(self._events)


__all__ = ["Timeline", "TimelineEvent"]
