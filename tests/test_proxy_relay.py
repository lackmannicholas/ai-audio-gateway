"""Phase 2 proof: the proxy relay works end-to-end over real gRPC, for both
agents, with no realtime model in the picture.

We stand up the real business gRPC server, connect the real gateway client, and
drive tool calls through the proxies. This isolates and proves the central
pattern: the gateway holds hollow proxies, execution happens in the business
plane, results correlate back by tool_call_id.
"""

from __future__ import annotations

import asyncio

import pytest

from business.grpc_server import BusinessBridgeServer
from gateway.grpc_business_client import BusinessBridgeClient
from proto_contract.auth import auth_metadata

import grpc


async def _connect(bind: str) -> tuple[BusinessBridgeServer, grpc.aio.Channel, BusinessBridgeClient]:
    server = BusinessBridgeServer()
    await server.start(bind=bind)  # insecure for tests
    channel = grpc.aio.insecure_channel(bind)
    client = BusinessBridgeClient(channel, metadata=auth_metadata())
    await client.open()
    return server, channel, client


@pytest.mark.asyncio
async def test_single_agent_each_tool_crosses_the_wire():
    server, channel, client = await _connect("127.0.0.1:8101")
    try:
        cfg = await client.start_session("cafe_single")
        # Single agent exposes all four café tools as proxies.
        assert set(cfg.proxies) == {"get_menu", "get_store_hours",
                                    "place_order", "check_order_status"}

        # Call a proxy -> relays across the wire -> executes in business plane.
        menu = await cfg.proxies["get_menu"].call({})
        assert any(item["name"] == "latte" for item in menu)

        order = await cfg.proxies["place_order"].call(
            {"items": [{"drink": "latte", "size": "large", "milk": "oat"}]})
        assert order["ok"] and order["total"] == 5.25
    finally:
        await client.end_call()
        await channel.close()
        await server.stop()


@pytest.mark.asyncio
async def test_responder_thinker_one_proxy_nested_fanout():
    server, channel, client = await _connect("127.0.0.1:8102")
    local_calls: list[str] = []
    client.on_local_tool_call = local_calls.append
    try:
        cfg = await client.start_session("cafe_responder_thinker")
        # Responder/thinker exposes exactly ONE proxy: the thinker.
        assert list(cfg.proxies) == ["consult_thinker"]

        # One wire call...
        answer = await cfg.proxies["consult_thinker"].call(
            {"request": "I'll have a large oat latte"})
        assert "order" in answer["answer"].lower()

        # ...but the business plane fanned out into local tool calls the gateway
        # only saw as informational notifications, never as wire tool calls.
        await asyncio.sleep(0.05)  # let notifications drain
        assert "get_menu" in local_calls
        assert "place_order" in local_calls
    finally:
        await client.end_call()
        await channel.close()
        await server.stop()


@pytest.mark.asyncio
async def test_same_gateway_client_drives_both_agents():
    """The gateway code path is identical; only the agent_kind differs."""
    server = BusinessBridgeServer()
    await server.start(bind="127.0.0.1:8103")
    try:
        for kind, expected in [("cafe_single", 4), ("cafe_responder_thinker", 1)]:
            channel = grpc.aio.insecure_channel("127.0.0.1:8103")
            client = BusinessBridgeClient(channel, metadata=auth_metadata())
            await client.open()
            cfg = await client.start_session(kind)
            assert len(cfg.proxies) == expected
            await client.end_call()
            await channel.close()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_bad_token_is_rejected():
    server = BusinessBridgeServer()
    await server.start(bind="127.0.0.1:8104")
    try:
        channel = grpc.aio.insecure_channel("127.0.0.1:8104")
        client = BusinessBridgeClient(channel, metadata=[("x-bridge-token", "wrong")])
        await client.open()
        with pytest.raises(Exception):
            await asyncio.wait_for(client.start_session("cafe_single"), timeout=3)
        await channel.close()
    finally:
        await server.stop()
