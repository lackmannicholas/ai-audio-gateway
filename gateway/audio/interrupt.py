"""Turn + interrupt tracking — the gateway's source of truth for turn_id.

The gateway owns the notion of "which turn are we on." Every time the caller
barges in, the turn_id increments. That number rides along on tool-call requests
to the business plane, and the business plane snapshots it before slow work
(the thinker) and compares after. If it changed, the slow work is stale and its
result is discarded rather than spoken over a conversation that has moved on.

This is the media-plane half of the staleness mechanism. The business-plane half
lives in the thinker. They coordinate purely through the turn_id on the wire —
no shared memory, because they are different processes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InterruptState:
    turn_id: int = 0
    response_active: bool = False

    def begin_response(self) -> None:
        self.response_active = True

    def end_response(self) -> None:
        self.response_active = False

    def barge_in(self) -> int:
        """Caller interrupted. Bump the turn and return the new id."""
        self.turn_id += 1
        self.response_active = False
        return self.turn_id


__all__ = ["InterruptState"]
