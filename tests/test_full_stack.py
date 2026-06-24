"""Phase 6 proof: the full stack runs end-to-end with the mock realtime model,
for both agents, and the gateway code path is identical for each.

mock realtime -> gateway session -> proxy relay -> business plane -> back.
We assert on the UI event stream the session emits, which is exactly what the
browser would render.
"""

from __future__ import annotations

import asyncio

import grpc
import pytest

from business.grpc_server import BusinessBridgeServer
from gateway.realtime.mock_backend import MockRealtimeBackend
from gateway.session import GatewaySession


import pytest


@pytest.fixture(autouse=True)
def _force_insecure_bridge(monkeypatch):
    """Scope insecure bridge to this module only, so the dedicated mTLS test
    (which needs cert auto-detection) isn't affected by a leaked env var."""
    monkeypatch.setenv("BRIDGE_INSECURE", "1")


class UICollector:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def __call__(self, kind: str, data: dict) -> None:
        self.events.append((kind, data))

    def kinds(self) -> list[str]:
        return [k for k, _ in self.events]

    def names_for(self, kind: str) -> list[str]:
        return [d.get("name") for k, d in self.events if k == kind]


async def _run_one_turn(agent_kind: str, bind: str) -> UICollector:
    server = BusinessBridgeServer()
    await server.start(bind=bind)
    ui = UICollector()
    try:
        session = GatewaySession(
            realtime=MockRealtimeBackend(),
            business_addr=bind,
            agent_kind=agent_kind,
            ui=ui,
        )
        await session.start()
        # The mock produces a full turn when we commit an utterance. Drive the
        # VAD through a real start->stop: a few loud frames, then enough silence
        # frames to cross the stop threshold (which triggers commit_audio).
        loud = (b"\x00\x40" * 160)   # RMS ~16384, well above threshold
        quiet = (b"\x00\x00" * 160)  # silence
        for _ in range(3):
            await session.on_caller_audio(loud)
        for _ in range(12):
            await session.on_caller_audio(quiet)
        # Give the turn time to run through tool relay + audio.
        await asyncio.sleep(1.2)
        await session.stop()
    finally:
        await server.stop()
    return ui


@pytest.mark.asyncio
async def test_single_agent_full_turn():
    ui = await _run_one_turn("cafe_single", "127.0.0.1:8201")
    kinds = ui.kinds()
    assert "session_configured" in kinds
    # Single agent: tool calls cross the wire (get_menu, place_order).
    assert "get_menu" in ui.names_for("tool_call_requested")
    assert "place_order" in ui.names_for("tool_call_requested")
    assert "audio_delta" in kinds  # assistant spoke


@pytest.mark.asyncio
async def test_responder_thinker_full_turn():
    ui = await _run_one_turn("cafe_responder_thinker", "127.0.0.1:8202")
    kinds = ui.kinds()
    assert "session_configured" in kinds
    # RT: ONE tool call crosses the wire (consult_thinker)...
    assert ui.names_for("tool_call_requested") == ["consult_thinker"]
    # ...but the nested café tools surface as LOCAL calls (never crossed wire).
    local = ui.names_for("local_tool_call")
    assert "get_menu" in local
    assert "place_order" in local
    assert "audio_delta" in kinds
