"""Run the business plane gRPC server.

    python -m business.server         # insecure (no certs present)
    python -m business.server         # mTLS if harness/certs/*.crt exist

Listens on 127.0.0.1:8002 by default (override with BUSINESS_ADDR).
"""

from __future__ import annotations

import asyncio
import logging
import os

from business.grpc_server import BusinessBridgeServer
from proto_contract.auth import server_credentials
from proto_contract.env import load_runtime_env

load_runtime_env()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    addr = os.getenv("BUSINESS_ADDR", "127.0.0.1:8002")
    server = BusinessBridgeServer()
    await server.start(bind=addr, server_credentials=server_credentials())
    await server.wait()


if __name__ == "__main__":
    asyncio.run(main())
