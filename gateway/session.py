"""The gateway session: the conductor of the media plane.

It owns one call's worth of coordination:
  * opens the bridge to the business plane and gets the session config (prompt +
    proxy tools),
  * configures the realtime backend with that prompt + the proxy tool schemas,
  * pumps caller audio -> VAD -> realtime backend,
  * pumps realtime events: relays tool calls through the proxies (across the
    wire), forwards audio out, tracks turns,
  * on caller barge-in: cancels the response and bumps turn_id (which the
    business plane uses to invalidate stale thinker work).

It never blocks audio on business work. Tool relays are awaited off to the side;
audio frames keep flowing.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import grpc

from gateway.audio.interrupt import InterruptState
from gateway.audio.vad import EnergyVAD
from gateway.grpc_business_client import BusinessBridgeClient
from gateway.realtime.protocol import (
    RealtimeBackend,
    RealtimeEvent,
    RealtimeEventType,
    RealtimeSessionConfig,
)
from proto_contract.auth import auth_metadata, channel_credentials

logger = logging.getLogger("gateway.session")

# A UI sink: the session emits structured updates the web layer streams to the
# browser. Signature: (kind, data) -> None.
UISink = Callable[[str, dict], None]


class GatewaySession:
    def __init__(
        self,
        realtime: RealtimeBackend,
        business_addr: str = "127.0.0.1:8002",
        agent_kind: str = "cafe_single",
        ui: UISink | None = None,
    ) -> None:
        self._realtime = realtime
        self._business_addr = business_addr
        self._agent_kind = agent_kind
        self._ui = ui or (lambda kind, data: None)

        self._vad = EnergyVAD()
        self._interrupt = InterruptState()
        self._client: BusinessBridgeClient | None = None
        self._channel: grpc.aio.Channel | None = None
        self._proxies: dict = {}
        self._event_task: asyncio.Task | None = None

    # -- lifecycle ----------------------------------------------------------- #
    async def start(self) -> None:
        creds = channel_credentials()
        if creds is not None:
            self._channel = grpc.aio.secure_channel(self._business_addr, creds)
            logger.info("gateway -> business over mTLS")
        else:
            self._channel = grpc.aio.insecure_channel(self._business_addr)
            logger.info("gateway -> business insecure (no certs found)")

        self._client = BusinessBridgeClient(self._channel, metadata=auth_metadata())
        self._client.on_local_tool_call = lambda name: self._ui(
            "local_tool_call", {"name": name})
        await self._client.open()

        cfg = await self._client.start_session(self._agent_kind)
        self._proxies = cfg.proxies
        self._ui("session_configured",
                 {"agent": cfg.agent_name, "tools": list(cfg.proxies)})

        await self._realtime.configure(RealtimeSessionConfig(
            instructions=cfg.instructions,
            greeting_instructions=cfg.greeting_instructions,
            tools=[p.spec for p in cfg.proxies.values()],
        ))

        self._event_task = asyncio.create_task(self._pump_realtime_events())

    # -- inbound caller audio ------------------------------------------------ #
    async def on_caller_audio(self, pcm: bytes) -> None:
        verdict = self._vad.process(pcm)
        if verdict == "start":
            # Caller started talking. If the assistant is mid-response, barge-in.
            if self._interrupt.response_active:
                await self._barge_in()
            self._ui("user_speech_started", {})
        elif verdict == "stop":
            self._ui("user_speech_stopped", {})
            await self._realtime.commit_audio()
        await self._realtime.append_audio(pcm)

    async def _barge_in(self) -> None:
        new_turn = self._interrupt.barge_in()
        await self._realtime.cancel_response()
        if self._client is not None:
            await self._client.barge_in()  # bumps the wire turn_id too
        self._ui("barge_in", {"turn_id": new_turn})
        logger.info("barge-in -> turn %d", new_turn)

    # -- outbound realtime events -------------------------------------------- #
    async def _pump_realtime_events(self) -> None:
        async for ev in self._realtime.events():
            await self._handle_realtime_event(ev)

    async def _handle_realtime_event(self, ev: RealtimeEvent) -> None:
        if ev.type is RealtimeEventType.TOOL_CALL:
            # The model called a (proxy) tool. Relay across the wire — but do it
            # off to the side so audio handling never blocks on it.
            asyncio.create_task(self._relay_tool(ev))

        elif ev.type is RealtimeEventType.AUDIO_DELTA:
            self._interrupt.begin_response()
            self._ui("audio_delta", {"pcm_len": ev.payload.get("pcm_len", 0)})

        elif ev.type is RealtimeEventType.TRANSCRIPT:
            self._ui("transcript", {"role": ev.payload.get("role"),
                                    "text": ev.payload.get("text", "")})

        elif ev.type is RealtimeEventType.RESPONSE_DONE:
            self._interrupt.end_response()
            self._ui("response_done", {})

        elif ev.type is RealtimeEventType.SPEECH_STARTED:
            if self._interrupt.response_active:
                await self._barge_in()

    async def _relay_tool(self, ev: RealtimeEvent) -> None:
        name = ev.payload.get("name")
        tool_call_id = ev.payload.get("tool_call_id")
        import json
        args = json.loads(ev.payload.get("arguments_json") or "{}")
        self._ui("tool_call_requested", {"name": name})
        proxy = self._proxies.get(name)
        if proxy is None:
            await self._realtime.submit_tool_output(tool_call_id, "null")
            return
        # The proxy call IS the relay across the wire to the business plane.
        result = await proxy.call(args)
        await self._realtime.submit_tool_output(tool_call_id, json.dumps(result))
        self._ui("tool_call_output", {"name": name})

    async def stop(self) -> None:
        if self._event_task:
            self._event_task.cancel()
        if self._client:
            await self._client.end_call()
        await self._realtime.close()
        if self._channel:
            await self._channel.close()


__all__ = ["GatewaySession", "UISink"]
