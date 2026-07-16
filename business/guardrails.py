"""Guardrails — safety checks that run in the business plane.

This is the payoff of the two-plane split for safety: the rule about what the
assistant may say lives with *meaning*, not in the media plane. The gateway just
forwards transcripts across the wire; the business plane evaluates them and, on a
violation, sends a ``response.cancel`` command back. The gateway enforces it by
cancelling the realtime response and clearing the audio queues — it never has to
know *why*, only that it was told to stop.

Keeping the policy here (rather than in the gateway or baked into the realtime
model's prompt) means it is centralized, testable, and swappable without touching
the hot path — and it applies identically no matter which agent or realtime model
is behind it.

The reference guardrail is deliberately trivial — a blocked-topic substring match
(default: "weather") — so the *mechanism* is the lesson, not the classifier. A
real deployment would swap ``BlockedTopicGuardrail`` for a moderation model, a
PII detector, a policy LLM, etc. The interface is the same: given a transcript,
allow or block.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


@dataclass
class GuardrailVerdict:
    allowed: bool
    rule: str | None = None
    reason: str | None = None


class Guardrail(Protocol):
    name: str

    def check(self, role: str, text: str) -> GuardrailVerdict:
        """Evaluate one transcript (``role`` is 'user' or 'assistant')."""
        ...


class BlockedTopicGuardrail:
    """Blocks a turn whose transcript mentions any configured topic.

    Substring, case-insensitive — intentionally the simplest thing that proves
    the cross-wire cancel loop. Applies to both the caller's words and the
    assistant's, so it covers "don't let the agent discuss X" and "don't let the
    caller steer us onto X" with one rule.
    """

    name = "blocked_topic"

    def __init__(self, topics: list[str]) -> None:
        self._topics = [t.lower() for t in topics if t.strip()]

    def check(self, role: str, text: str) -> GuardrailVerdict:
        low = (text or "").lower()
        for topic in self._topics:
            if topic in low:
                return GuardrailVerdict(
                    allowed=False,
                    rule=self.name,
                    reason=f"{role} transcript mentioned a blocked topic: {topic!r}",
                )
        return GuardrailVerdict(allowed=True)


class GuardrailSet:
    """An ordered set of guardrails; the first violation wins."""

    def __init__(self, guardrails: list[Guardrail]) -> None:
        self._guardrails = guardrails

    def __bool__(self) -> bool:
        return bool(self._guardrails)

    def evaluate(self, role: str, text: str) -> GuardrailVerdict:
        for guardrail in self._guardrails:
            verdict = guardrail.check(role, text)
            if not verdict.allowed:
                return verdict
        return GuardrailVerdict(allowed=True)


def build_guardrails() -> GuardrailSet:
    """Build the session's guardrails from env.

    ``GUARDRAIL_BLOCKED_TOPICS`` is a comma-separated list; default "weather".
    Set it empty to disable.
    """
    raw = os.getenv("GUARDRAIL_BLOCKED_TOPICS", "weather")
    topics = [t.strip() for t in raw.split(",") if t.strip()]
    guardrails: list[Guardrail] = []
    if topics:
        guardrails.append(BlockedTopicGuardrail(topics))
    return GuardrailSet(guardrails)


__all__ = [
    "Guardrail",
    "GuardrailVerdict",
    "GuardrailSet",
    "BlockedTopicGuardrail",
    "build_guardrails",
]
