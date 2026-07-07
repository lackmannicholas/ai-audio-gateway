"""Run the audio gateway (browser-facing ASGI app).

    python -m gateway.server

Serves the demo UI at http://127.0.0.1:8001 and the live websocket at /ws.
Connects to the business plane at BUSINESS_ADDR (default 127.0.0.1:8002).

Realtime backend is chosen by REALTIME_BACKEND (mock | openai); mock by default
so this runs with no API key.
"""

from __future__ import annotations

import os

import uvicorn

from gateway.app import build_app
from proto_contract.env import load_runtime_env

load_runtime_env()

app = build_app(business_addr=os.getenv("BUSINESS_ADDR", "127.0.0.1:8002"))


def main() -> None:
    # Default to loopback for local runs; containers must set GATEWAY_HOST=0.0.0.0
    # (compose does) or the published port maps to nothing.
    uvicorn.run(
        app,
        host=os.getenv("GATEWAY_HOST", "127.0.0.1"),
        port=int(os.getenv("GATEWAY_PORT", "8001")),
    )


if __name__ == "__main__":
    main()
