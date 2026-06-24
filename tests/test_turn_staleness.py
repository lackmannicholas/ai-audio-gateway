"""Phase 5 proof: cross-wire turn staleness.

The hardest, most article-worthy mechanic. The thinker runs in the business
plane and takes real time. If the caller barges in while it's working, the
gateway bumps turn_id and sends a barge_in event. The business plane records the
new turn, which makes the in-flight thinker run stale — so instead of returning
an answer that would be spoken over a conversation that has moved on, the thinker
abandons the work and reports staleness.

Two processes, coordinating real-time state through nothing but the turn_id on
the wire.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import grpc
import pytest

from business.grpc_server import BusinessBridgeServer
from business.thinker import MockThinkerModel, ModelStep
from gateway.grpc_business_client import BusinessBridgeClient
from gateway.realtime.protocol import (
    RealtimeBackend,
    RealtimeEvent,
    RealtimeEventType,
    RealtimeSessionConfig,
)
from gateway.session import GatewaySession
from proto_contract.auth import auth_metadata


class SlowThinkerModel(MockThinkerModel):
    """A thinker model whose first step blocks, so a barge-in can land mid-run."""

    def __init__(self, gate: asyncio.Event) -> None:
        self._gate = gate

    async def step(self, request: str, history):
        if not history:
            # Block until the test releases us (simulating slow reasoning).
            await self._gate.wait()
        return await super().step(request, history)


class ProbeRealtimeBackend(RealtimeBackend):
    """Realtime probe that records whether stale tool output is submitted."""

    def __init__(self) -> None:
        self.out: asyncio.Queue[RealtimeEvent] = asyncio.Queue()
        self.outputs: list[tuple[str, dict]] = []
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

    async def submit_tool_output(self, tool_call_id: str, output_json: str) -> None:
        self.outputs.append((tool_call_id, json.loads(output_json)))

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        while True:
            yield await self.out.get()


@pytest.mark.asyncio
async def test_barge_in_makes_thinker_result_stale():
    # Patch the responder/thinker agent to use the slow model.
    import business.agents.responder_thinker as rt_mod

    gate = asyncio.Event()
    original_init = rt_mod.ResponderThinkerAgent.__init__

    def patched_init(self, thinker_model=None):
        original_init(self, thinker_model=SlowThinkerModel(gate))
    rt_mod.ResponderThinkerAgent.__init__ = patched_init

    server = BusinessBridgeServer()
    await server.start(bind="127.0.0.1:8105")
    try:
        channel = grpc.aio.insecure_channel("127.0.0.1:8105")
        client = BusinessBridgeClient(channel, metadata=auth_metadata())
        await client.open()
        cfg = await client.start_session("cafe_responder_thinker")

        # Kick off the (slow) thinker call on turn 0.
        call_task = asyncio.create_task(
            cfg.proxies["consult_thinker"].call({"request": "large oat latte"}))
        await asyncio.sleep(0.1)  # ensure the thinker is mid-run, blocked on gate

        # Caller barges in: gateway bumps turn_id and notifies the business plane.
        new_turn = await client.barge_in()
        assert new_turn == 1
        await asyncio.sleep(0.05)  # let the barge_in event arrive at the server

        # Release the thinker. It should now detect staleness (turn moved 0 -> 1)
        # and abandon rather than return an answer.
        gate.set()
        result = await asyncio.wait_for(call_task, timeout=3)
        assert result.get("stale") is True
        assert result.get("turn_id") == 0

        await client.end_call()
        await channel.close()
    finally:
        rt_mod.ResponderThinkerAgent.__init__ = original_init
        await server.stop()


@pytest.mark.asyncio
async def test_gateway_does_not_submit_stale_tool_output_after_barge_in(monkeypatch):
    """A stale thinker result must not wake the realtime model back up."""
    monkeypatch.setenv("BRIDGE_INSECURE", "1")

    import business.agents.responder_thinker as rt_mod

    gate = asyncio.Event()
    original_init = rt_mod.ResponderThinkerAgent.__init__

    def patched_init(self, thinker_model=None):
        original_init(self, thinker_model=SlowThinkerModel(gate))
    rt_mod.ResponderThinkerAgent.__init__ = patched_init

    server = BusinessBridgeServer()
    await server.start(bind="127.0.0.1:8107")
    backend = ProbeRealtimeBackend()
    ui: list[tuple[str, dict]] = []
    session = GatewaySession(
        realtime=backend,
        business_addr="127.0.0.1:8107",
        agent_kind="cafe_responder_thinker",
        ui=lambda kind, data: ui.append((kind, data)),
    )
    try:
        await session.start()
        await backend.out.put(RealtimeEvent(
            RealtimeEventType.TOOL_CALL,
            {
                "name": "consult_thinker",
                "tool_call_id": "probe_tc",
                "arguments_json": json.dumps({"request": "large oat latte"}),
            },
        ))
        await asyncio.sleep(0.1)  # ensure the thinker is blocked mid-run

        await session._barge_in()  # noqa: SLF001 - exercise the same demo hook
        await asyncio.sleep(0.05)  # let barge_in reach the business plane
        gate.set()

        for _ in range(20):
            if any(kind == "tool_call_stale" for kind, _ in ui):
                break
            await asyncio.sleep(0.05)

        assert backend.cancelled == 1
        assert backend.outputs == []
        assert any(kind == "tool_call_stale" for kind, _ in ui)
        assert not any(kind == "tool_call_output" for kind, _ in ui)
    finally:
        await session.stop()
        rt_mod.ResponderThinkerAgent.__init__ = original_init
        await server.stop()


@pytest.mark.asyncio
async def test_no_barge_in_returns_answer():
    """Control: without a barge-in, the same call returns a real answer."""
    server = BusinessBridgeServer()
    await server.start(bind="127.0.0.1:8106")
    try:
        channel = grpc.aio.insecure_channel("127.0.0.1:8106")
        client = BusinessBridgeClient(channel, metadata=auth_metadata())
        await client.open()
        cfg = await client.start_session("cafe_responder_thinker")
        result = await cfg.proxies["consult_thinker"].call({"request": "large oat latte"})
        assert "answer" in result
        assert "stale" not in result
        await client.end_call()
        await channel.close()
    finally:
        await server.stop()
