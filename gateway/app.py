"""The gateway's ASGI app: the browser-facing edge.

Exposes:
  * GET  /              -> the demo UI (served from harness/browser)
  * WS   /ws            -> the live session: browser sends control + audio
                          frames; gateway streams back UI events (the same
                          structured events the GatewaySession emits).

This plays the role Twilio's media-stream websocket plays in production, minus
the PSTN. The browser is the "phone." Audio frames in, UI events out. The agent
(single vs responder/thinker) is chosen by the browser when it connects, which
is how the same gateway drives either one.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from gateway.realtime.openai_backend import build_realtime_backend
from gateway.rtc import RTCGateway, rtc_barge_in, rtc_disconnect, rtc_events, rtc_offer
from gateway.session import GatewaySession
from proto_contract.env import load_runtime_env

load_runtime_env()

logger = logging.getLogger("gateway.app")

_BROWSER_DIR = Path(__file__).resolve().parent.parent / "harness" / "browser"


async def _index(request: Request) -> FileResponse:
    return FileResponse(_BROWSER_DIR / "index.html")


async def _health(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


async def _favicon(request: Request) -> Response:
    return Response(status_code=204)


async def _ws(websocket: WebSocket) -> None:
    await websocket.accept()
    agent_kind = websocket.query_params.get("agent", "cafe_single")
    business_addr = websocket.app.state.business_addr

    loop = asyncio.get_running_loop()

    def ui_sink(kind: str, data: dict) -> None:
        # Fan UI events to the browser. Scheduled threadsafe-ish since the
        # session may emit from tasks on this same loop.
        asyncio.ensure_future(websocket.send_json({"kind": kind, **data}))

    session = GatewaySession(
        realtime=build_realtime_backend(),
        business_addr=business_addr,
        agent_kind=agent_kind,
        ui=ui_sink,
    )
    await session.start()
    await websocket.send_json({"kind": "connected", "agent": agent_kind})

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            text = msg.get("text")
            if text is None:
                continue
            frame = json.loads(text)
            kind = frame.get("kind")
            if kind == "audio":
                pcm = base64.b64decode(frame.get("pcm_b64", ""))
                await session.on_caller_audio(pcm)
            elif kind == "barge_in":
                # Explicit barge from the UI button (demo affordance).
                await session._barge_in()  # noqa: SLF001 - demo hook
            elif kind == "hangup":
                break
    except WebSocketDisconnect:
        pass
    finally:
        await session.stop()


def build_app(business_addr: str = "127.0.0.1:8002") -> Starlette:
    app = Starlette(routes=[
        Route("/", _index),
        Route("/health", _health),
        Route("/favicon.ico", _favicon),
        Route("/api/rtc/offer", rtc_offer, methods=["POST"]),
        Route("/api/rtc/barge-in", rtc_barge_in, methods=["POST"]),
        Route("/api/rtc/disconnect", rtc_disconnect, methods=["POST"]),
        Route("/api/events/{session_id}", rtc_events),
        WebSocketRoute("/ws", _ws),
        # Static assets (app.js, etc.)
    ])
    app.state.business_addr = business_addr
    app.state.rtc_gateway = RTCGateway()
    app.mount("/static", StaticFiles(directory=str(_BROWSER_DIR)), name="static")
    return app


__all__ = ["build_app"]
