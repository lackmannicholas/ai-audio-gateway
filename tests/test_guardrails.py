"""Guardrails run in the business plane; enforcement happens in the gateway.

Two levels of proof:
  * unit — the blocked-topic guardrail allows/blocks the right transcripts;
  * end to end — a banned transcript pushed from the gateway makes the business
    plane emit response.cancel, and the gateway cancels the realtime response
    and clears audio. That round trip across two processes is the point.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import grpc
import pytest

from business.grpc_server import BusinessBridgeServer
from business.guardrails import BlockedTopicGuardrail, GuardrailSet, build_guardrails
from gateway.realtime.protocol import (
    RealtimeBackend,
    RealtimeEvent,
    RealtimeEventType,
    RealtimeSessionConfig,
)
from gateway.session import GatewaySession


# --------------------------------------------------------------------------- #
# Unit
# --------------------------------------------------------------------------- #
def test_blocked_topic_blocks_and_allows():
    g = BlockedTopicGuardrail(["weather"])
    assert g.check("assistant", "It's sunny — great WEATHER today").allowed is False
    assert g.check("user", "I'll have a large oat latte").allowed is True


def test_guardrail_set_first_violation_wins():
    gs = GuardrailSet([BlockedTopicGuardrail(["weather", "politics"])])
    verdict = gs.evaluate("assistant", "let's talk politics")
    assert verdict.allowed is False
    assert verdict.rule == "blocked_topic"
    assert "politics" in verdict.reason


def test_build_guardrails_env(monkeypatch):
    monkeypatch.setenv("GUARDRAIL_BLOCKED_TOPICS", "weather, sports")
    gs = build_guardrails()
    assert gs.evaluate("user", "how about them sports").allowed is False
    assert gs.evaluate("user", "a flat white please").allowed is True

    monkeypatch.setenv("GUARDRAIL_BLOCKED_TOPICS", "")
    assert not build_guardrails()  # disabled -> empty set is falsy


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
class ProbeRealtimeBackend(RealtimeBackend):
    """Records cancels and lets the test inject model events (transcripts)."""

    def __init__(self) -> None:
        self.out: asyncio.Queue[RealtimeEvent] = asyncio.Queue()
        self.cancelled = 0

    async def configure(self, config: RealtimeSessionConfig) -> None:
        return None

    async def append_audio(self, pcm: bytes) -> None:
        return None

    async def commit_audio(self) -> None:
        return None

    async def create_response(self, instructions: str | None = None) -> None:
        return None

    async def cancel_response(self) -> None:
        self.cancelled += 1

    async def submit_tool_output(self, tool_call_id: str, output_json: str,
                                 create_response: bool = True) -> None:
        return None

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        while True:
            yield await self.out.get()


async def _run_with_transcript(text: str, bind: str, monkeypatch) -> tuple[int, list]:
    monkeypatch.setenv("BRIDGE_INSECURE", "1")
    monkeypatch.setenv("GUARDRAIL_BLOCKED_TOPICS", "weather")

    server = BusinessBridgeServer()
    await server.start(bind=bind)
    backend = ProbeRealtimeBackend()
    ui: list[tuple[str, dict]] = []
    session = GatewaySession(
        realtime=backend,
        business_addr=bind,
        agent_kind="cafe_single",
        ui=lambda kind, data: ui.append((kind, data)),
    )
    try:
        await session.start()
        # The model produces an assistant transcript; the gateway forwards it
        # across the wire for guardrailing.
        await backend.out.put(RealtimeEvent(
            RealtimeEventType.TRANSCRIPT, {"role": "assistant", "text": text}))
        for _ in range(40):
            if any(k == "guardrail_blocked" for k, _ in ui):
                break
            await asyncio.sleep(0.05)
    finally:
        await session.stop()
        await server.stop()
    return backend.cancelled, ui


@pytest.mark.asyncio
async def test_banned_transcript_cancels_across_the_wire(monkeypatch):
    cancelled, ui = await _run_with_transcript(
        "Sure — the weather today is sunny.", "127.0.0.1:8231", monkeypatch)
    assert cancelled == 1
    blocked = [d for k, d in ui if k == "guardrail_blocked"]
    assert blocked and blocked[0]["rule"] == "blocked_topic"
    assert "weather" in blocked[0]["reason"]


@pytest.mark.asyncio
async def test_clean_transcript_is_not_cancelled(monkeypatch):
    cancelled, ui = await _run_with_transcript(
        "Your large oat latte is $5.25.", "127.0.0.1:8232", monkeypatch)
    assert cancelled == 0
    assert not any(k == "guardrail_blocked" for k, _ in ui)
