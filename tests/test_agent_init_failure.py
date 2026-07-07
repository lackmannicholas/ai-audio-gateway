"""Agent init failure must surface, not hang.

In OpenAI mode the responder-thinker agent constructs an OpenAI-backed thinker;
if the `openai` extra isn't installed (or the model can't be built), that raises
while the business plane is handling call.started. If the error propagates
silently the gateway waits forever for session.configure and the connect hangs.
The business plane must instead report an ERROR so the gateway fails fast.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress

import grpc
import pytest

import business.grpc_server as grpc_server
from business.grpc_server import BusinessBridgeServer
from gateway.grpc_business_client import BusinessBridgeClient
from proto_contract.auth import auth_metadata


@pytest.mark.asyncio
async def test_start_session_raises_clear_error_when_agent_init_fails(monkeypatch):
    monkeypatch.setenv("BRIDGE_INSECURE", "1")

    def boom(agent_kind: str):
        raise ImportError("No module named 'openai'")

    monkeypatch.setattr(grpc_server, "_select_agent", boom)

    server = BusinessBridgeServer()
    await server.start(bind="127.0.0.1:8221")
    try:
        channel = grpc.aio.insecure_channel("127.0.0.1:8221")
        client = BusinessBridgeClient(channel, metadata=auth_metadata())
        await client.open()
        with pytest.raises(RuntimeError) as excinfo:
            # Must not hang: a bounded wait proves the error propagates.
            await asyncio.wait_for(
                client.start_session("cafe_responder_thinker"), timeout=5)
        assert "agent init failed" in str(excinfo.value)
        assert "openai" in str(excinfo.value)
        # The business plane closed the stream after reporting the error, so
        # end_call may find it gone — that's fine, we only care it didn't hang.
        with suppress(Exception):
            await client.end_call()
        await channel.close()
    finally:
        await server.stop()
