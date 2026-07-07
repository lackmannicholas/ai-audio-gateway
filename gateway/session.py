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
import json
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress

import grpc

from gateway.audio.interrupt import InterruptState
from gateway.audio.vad import LocalVAD, PassthroughVAD, build_local_vad
from gateway.grpc_business_client import BusinessBridgeClient
from gateway.realtime.protocol import (
    RealtimeBackend,
    RealtimeBackendClosed,
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
        vad: LocalVAD | None = None,
        pending_audio: Callable[[], bool] | None = None,
    ) -> None:
        self._realtime = realtime
        self._business_addr = business_addr
        self._agent_kind = agent_kind
        self._ui = ui or (lambda kind, data: None)
        # Predicate: is there assistant audio still queued for playback? Set by
        # the transport (rtc.py) to the output track's state. The realtime model
        # streams audio faster than realtime, so a response finishes generating
        # (response_active flips false) while seconds of it are still queued.
        # Barge-in must fire during that tail too, or the interrupt is processed
        # while stale audio keeps playing.
        self._pending_audio = pending_audio

        self._vad = vad or build_local_vad()
        self._interrupt = InterruptState()
        self._client: BusinessBridgeClient | None = None
        self._channel: grpc.aio.Channel | None = None
        self._proxies: dict = {}
        self._event_task: asyncio.Task | None = None
        self._relay_tasks: set[asyncio.Task] = set()
        self._stopped = False
        # Turn-latency probe: stamped when the caller's utterance is committed,
        # cleared when the first assistant audio arrives. The delta is the
        # number a performance discussion actually cares about.
        self._utterance_committed_at: float | None = None

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

        # If local VAD isn't actually gating (passthrough fallback), the
        # realtime backend must keep server-side turn detection on — otherwise
        # nothing commits the audio buffer or creates a response.
        local_endpointing = not isinstance(self._vad, PassthroughVAD)
        if not local_endpointing:
            logger.info(
                "local VAD not gating; leaving server-side turn detection on")
        await self._realtime.configure(RealtimeSessionConfig(
            instructions=cfg.instructions,
            greeting_instructions=cfg.greeting_instructions,
            tools=[p.spec for p in cfg.proxies.values()],
            server_vad=not local_endpointing,
        ))

        self._event_task = asyncio.create_task(self._pump_realtime_events())

    # -- inbound caller audio ------------------------------------------------ #
    def _assistant_holding_channel(self) -> bool:
        """The assistant occupies the channel while it is generating a response
        OR while audio is still queued for playback."""
        if self._interrupt.response_active:
            return True
        return bool(self._pending_audio and self._pending_audio())

    async def on_caller_audio(self, pcm: bytes) -> None:
        vad_result = self._vad.process(pcm)
        if vad_result.speech_started:
            # Caller started talking. If the assistant is still holding the
            # channel (generating or playing), barge-in.
            if self._assistant_holding_channel():
                await self._barge_in()
            self._ui("user_speech_started", {
                "speech_probability": vad_result.speech_probability,
            })

        for chunk in vad_result.frames_to_flush:
            await self._realtime.append_audio(chunk)

        if vad_result.speech_ended:
            self._ui("user_speech_stopped", {})
            await self._realtime.commit_audio()
            self._utterance_committed_at = time.monotonic()

    async def _barge_in(self) -> None:
        # InterruptState is the single source of truth for turn_id; the bridge
        # client just carries it onto the wire.
        new_turn = self._interrupt.barge_in()
        await self._realtime.cancel_response()
        if self._client is not None:
            await self._client.barge_in(new_turn)
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
            task = asyncio.create_task(self._relay_tool(ev))
            self._relay_tasks.add(task)
            task.add_done_callback(self._relay_tasks.discard)

        elif ev.type is RealtimeEventType.AUDIO_DELTA:
            if self._utterance_committed_at is not None:
                latency_ms = int((time.monotonic() - self._utterance_committed_at) * 1000)
                self._utterance_committed_at = None
                self._ui("turn_latency", {"ms": latency_ms})
                logger.info("turn latency (commit -> first audio): %dms", latency_ms)
            self._interrupt.begin_response()
            data = {"pcm_len": ev.payload.get("pcm_len", 0)}
            if "audio_b64" in ev.payload:
                data["audio_b64"] = ev.payload["audio_b64"]
            self._ui("audio_delta", data)

        elif ev.type is RealtimeEventType.TRANSCRIPT:
            self._ui("transcript", {"role": ev.payload.get("role"),
                                    "text": ev.payload.get("text", "")})

        elif ev.type is RealtimeEventType.RESPONSE_DONE:
            self._interrupt.end_response()
            self._ui("response_done", {})

        elif ev.type is RealtimeEventType.SPEECH_STARTED:
            if self._assistant_holding_channel():
                await self._barge_in()

        elif ev.type is RealtimeEventType.ERROR:
            self._ui("error", {
                "message": ev.payload.get("message", "Realtime backend closed"),
            })

    async def _relay_tool(self, ev: RealtimeEvent) -> None:
        name = ev.payload.get("name")
        tool_call_id = ev.payload.get("tool_call_id")
        args = json.loads(ev.payload.get("arguments_json") or "{}")
        self._ui("tool_call_requested", {"name": name})
        proxy = self._proxies.get(name)
        if proxy is None:
            await self._realtime.submit_tool_output(tool_call_id, "null")
            return
        # The proxy call IS the relay across the wire to the business plane.
        result = await proxy.call(args)
        if isinstance(result, dict) and result.get("stale") is True:
            # The business plane finished work for a turn the caller already
            # interrupted. Do not wake the realtime model back up with it —
            # but do resolve the pending function call, otherwise it dangles
            # in the model's conversation and can break the next response.
            await self._realtime.submit_tool_output(
                tool_call_id,
                json.dumps({"cancelled": True, "reason": "superseded by barge-in"}),
                create_response=False,
            )
            self._ui("tool_call_stale", {
                "name": name,
                "turn_id": result.get("turn_id"),
            })
            return
        await self._realtime.submit_tool_output(tool_call_id, json.dumps(result))
        self._ui("tool_call_output", {"name": name})

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        if self._event_task:
            self._event_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._event_task

        for task in list(self._relay_tasks):
            task.cancel()
        for task in list(self._relay_tasks):
            with suppress(asyncio.CancelledError):
                await task

        if self._client:
            with suppress(Exception):
                await self._client.end_call()
        with suppress(RealtimeBackendClosed, Exception):
            await self._realtime.close()
        if self._channel:
            with suppress(Exception):
                await self._channel.close()


__all__ = ["GatewaySession", "UISink"]
