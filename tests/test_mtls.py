"""mTLS path test. Runs only when local certs have been generated
(``make certs``); skipped otherwise so the suite stays green on a fresh clone.

Proves the bridge works over mutual TLS: the business plane presents a server
cert, the gateway presents a client cert, and each verifies the other against
the shared CA. Uses ``localhost`` to match the cert SAN.
"""

from __future__ import annotations

from pathlib import Path

import grpc
import pytest

from business.grpc_server import BusinessBridgeServer
from gateway.grpc_business_client import BusinessBridgeClient
from proto_contract.auth import (
    auth_metadata,
    channel_credentials,
    server_credentials,
)

_CERTS = Path("harness/certs")
_have_certs = (_CERTS / "ca.crt").exists() and (_CERTS / "server.crt").exists()

pytestmark = pytest.mark.skipif(not _have_certs, reason="run `make certs` to enable mTLS test")


@pytest.mark.asyncio
async def test_bridge_over_mtls():
    server = BusinessBridgeServer()
    # localhost (not 127.0.0.1) so the server cert SAN matches on verify.
    await server.start(bind="localhost:8302", server_credentials=server_credentials())
    try:
        creds = channel_credentials()
        assert creds is not None, "expected client credentials with certs present"
        channel = grpc.aio.secure_channel("localhost:8302", creds)
        client = BusinessBridgeClient(channel, metadata=auth_metadata())
        await client.open()
        cfg = await client.start_session("cafe_single")
        assert len(cfg.proxies) == 4
        # And a tool call relays fine over the encrypted, mutually-authed channel.
        menu = await cfg.proxies["get_menu"].call({})
        assert any(i["name"] == "latte" for i in menu)
        await client.end_call()
        await channel.close()
    finally:
        await server.stop()
