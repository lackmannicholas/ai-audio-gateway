"""Guardrails run in the business plane; enforcement happens in the gateway.

The production shape is *streaming*: the assistant transcript is pushed across
the wire delta-by-delta, the business plane accumulates it per response and
re-checks on every chunk, and a violation cancels the response mid-sentence —
before the banned content is fully spoken. Levels of proof:

  * unit — the blocked-topic guardrail and the session's delta accumulator
    (topic split across deltas, per-response reset, cancel-once);
  * end to end — streamed deltas make the business plane emit response.cancel,
    and the gateway cancels the realtime response + clears audio, mid-stream.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from business.agents.single_agent import SingleVoiceAgent
from business.grpc_server import BusinessBridgeServer, BusinessSession
from business.guardrails import BlockedTopicGuardrail, GuardrailSet, build_guardrails
from gateway.realtime.protocol import (
    RealtimeBackend,
    RealtimeEvent,
    RealtimeEventType,
    RealtimeSessionConfig,
)
from gateway.session import GatewaySession


# --------------------------------------------------------------------------- #
# Unit — the guardrail
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
# Unit — the streaming accumulator on the business session
# --------------------------------------------------------------------------- #
def test_delta_accumulator_catches_topic_split_across_deltas(monkeypatch):
    monkeypatch.setenv("GUARDRAIL_BLOCKED_TOPICS", "weather")
    session = BusinessSession("call_1", SingleVoiceAgent())

    assert session.check_assistant_delta("r1", "The wea").allowed is True
    # "The wea" + "ther is nice" -> "The weather is nice" -> blocked
    assert session.check_assistant_delta("r1", "ther is nice").allowed is False
    # Same response already flagged: later deltas are suppressed (no re-cancel).
    assert session.check_assistant_delta("r1", " and warm").allowed is True
    # A new response resets the buffer and the tripped flag.
    assert session.check_assistant_delta("r2", "the weather").allowed is False


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


async def _emit_delta(backend: ProbeRealtimeBackend, delta: str,
                      response_id: str = "r1") -> None:
    await backend.out.put(RealtimeEvent(
        RealtimeEventType.TRANSCRIPT_DELTA,
        {"role": "assistant", "delta": delta, "response_id": response_id}))


async def _wait_for(ui: list, kind: str, timeout: float = 2.0) -> bool:
    for _ in range(int(timeout / 0.02)):
        if any(k == kind for k, _ in ui):
            return True
        await asyncio.sleep(0.02)
    return False


async def _with_session(bind: str, monkeypatch, drive):
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
        await drive(backend, ui)
    finally:
        await session.stop()
        await server.stop()
    return backend, ui


@pytest.mark.asyncio
async def test_streaming_deltas_cancel_mid_sentence(monkeypatch):
    async def drive(backend, ui):
        await _emit_delta(backend, "Sure, ")
        await _emit_delta(backend, "the ")
        await asyncio.sleep(0.25)
        # Nothing blocked yet: "Sure, the " is clean.
        assert not any(k == "guardrail_blocked" for k, _ in ui)
        # The delta that completes the banned word trips it — mid-sentence.
        await _emit_delta(backend, "weather ")
        assert await _wait_for(ui, "guardrail_blocked")
        # Tail deltas for the same (now-cancelled) response must not re-cancel.
        await _emit_delta(backend, "is sunny.")
        await asyncio.sleep(0.2)

    backend, ui = await _with_session("127.0.0.1:8231", monkeypatch, drive)
    assert backend.cancelled == 1
    blocked = [d for k, d in ui if k == "guardrail_blocked"]
    assert len(blocked) == 1
    assert blocked[0]["rule"] == "blocked_topic"
    assert "weather" in blocked[0]["reason"]


@pytest.mark.asyncio
async def test_clean_stream_is_not_cancelled(monkeypatch):
    async def drive(backend, ui):
        for delta in ["Your ", "large ", "oat ", "latte ", "is ", "$5.25."]:
            await _emit_delta(backend, delta)
        await asyncio.sleep(0.4)

    backend, ui = await _with_session("127.0.0.1:8232", monkeypatch, drive)
    assert backend.cancelled == 0
    assert not any(k == "guardrail_blocked" for k, _ in ui)


@pytest.mark.asyncio
async def test_user_input_transcript_is_guardrailed(monkeypatch):
    async def drive(backend, ui):
        # The caller's finalized transcript is the input-guardrail path.
        await backend.out.put(RealtimeEvent(
            RealtimeEventType.TRANSCRIPT,
            {"role": "user", "text": "hey, what's the weather today?"}))
        assert await _wait_for(ui, "guardrail_blocked")

    backend, ui = await _with_session("127.0.0.1:8233", monkeypatch, drive)
    assert backend.cancelled == 1
