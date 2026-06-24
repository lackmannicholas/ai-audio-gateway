"""Browser WebRTC transport for the audio gateway.

This mirrors the reference responder-thinker repo's shape:

    browser <--WebRTC audio--> gateway <--internal backend adapter--> realtime

The browser-facing media path is WebRTC. UI events are still structured JSON,
streamed over SSE so the visualizer can render the same event vocabulary the
websocket harness used.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import av
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from gateway.audio.convert import AudioConverter, WEBRTC_FRAME_SAMPLES, WEBRTC_SAMPLE_RATE
from gateway.realtime.openai_backend import build_realtime_backend
from gateway.realtime.protocol import RealtimeBackendClosed
from gateway.session import GatewaySession

logger = logging.getLogger("gateway.rtc")


class AudioOutputStream(MediaStreamTrack):
    """Paced WebRTC audio track fed by realtime output PCM frames."""

    kind = "audio"

    _PTIME = 0.02
    _SILENCE = np.zeros(WEBRTC_FRAME_SAMPLES, dtype=np.int16)

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue[av.AudioFrame] = asyncio.Queue()
        self._buffer = np.empty(0, dtype=np.int16)
        self._pts = 0
        self._start: float | None = None
        self._frame_count = 0

    @property
    def has_queued_audio(self) -> bool:
        return not self._queue.empty() or len(self._buffer) > 0

    def clear(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._buffer = np.empty(0, dtype=np.int16)

    @staticmethod
    def _make_frame(samples: np.ndarray) -> av.AudioFrame:
        frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1),
            format="s16",
            layout="mono",
        )
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        return frame

    async def push_frame(self, frame: av.AudioFrame) -> None:
        audio = frame.to_ndarray().flatten().astype(np.int16)
        self._buffer = np.concatenate([self._buffer, audio])

        while len(self._buffer) >= WEBRTC_FRAME_SAMPLES:
            chunk = self._buffer[:WEBRTC_FRAME_SAMPLES]
            self._buffer = self._buffer[WEBRTC_FRAME_SAMPLES:]
            await self._queue.put(self._make_frame(chunk))

    async def recv(self) -> av.AudioFrame:
        loop = asyncio.get_event_loop()
        if self._start is None:
            self._start = loop.time()

        target = self._start + (self._frame_count + 1) * self._PTIME
        delay = target - loop.time()

        if delay > 0:
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=delay)
                remaining = target - loop.time()
                if remaining > 0:
                    await asyncio.sleep(remaining)
            except asyncio.TimeoutError:
                frame = self._make_frame(self._SILENCE)
        else:
            try:
                frame = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                frame = self._make_frame(self._SILENCE)

        self._frame_count += 1
        frame.pts = self._pts
        frame.time_base = Fraction(1, WEBRTC_SAMPLE_RATE)
        self._pts += frame.samples
        return frame


@dataclass
class RTCGatewayCall:
    session_id: str
    peer_connection: RTCPeerConnection
    business_addr: str
    agent_kind: str
    output_track: AudioOutputStream = field(default_factory=AudioOutputStream)
    on_closed: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        self.event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.converter = AudioConverter()
        self.gateway = GatewaySession(
            realtime=build_realtime_backend(),
            business_addr=self.business_addr,
            agent_kind=self.agent_kind,
            ui=self._emit_ui_event,
        )
        self._audio_task: asyncio.Task | None = None
        self._started = asyncio.Event()
        self._closed = False
        self._close_lock = asyncio.Lock()

    async def start(self) -> None:
        await self.gateway.start()
        self._started.set()
        self.event_queue.put_nowait({
            "kind": "connected",
            "agent": self.agent_kind,
            "transport": "webrtc",
            "mock_tts": os.getenv("REALTIME_BACKEND", "mock").lower() != "openai",
        })

    def _emit_ui_event(self, kind: str, data: dict) -> None:
        event_data = dict(data)
        if kind == "audio_delta":
            audio_b64 = event_data.pop("audio_b64", None)
            if audio_b64:
                asyncio.create_task(self._push_realtime_audio(str(audio_b64)))
        elif kind == "barge_in":
            self.output_track.clear()
        self.event_queue.put_nowait({"kind": kind, **event_data})

    async def _push_realtime_audio(self, audio_b64: str) -> None:
        frame = self.converter.realtime_b64_to_aiortc_frame(audio_b64)
        await self.output_track.push_frame(frame)

    def attach_input_track(self, track: MediaStreamTrack) -> None:
        self._audio_task = asyncio.create_task(self._consume_input_track(track))

    async def _consume_input_track(self, track: MediaStreamTrack) -> None:
        await self._started.wait()
        try:
            while not self._closed:
                frame = await track.recv()
                pcm = self.converter.aiortc_frame_to_pcm16(frame)
                if pcm:
                    await self.gateway.on_caller_audio(pcm)
        except (asyncio.CancelledError, MediaStreamError):
            pass
        except RealtimeBackendClosed as exc:
            logger.warning("realtime backend closed: %s", exc)
            self.event_queue.put_nowait({"kind": "error", "message": str(exc)})
        except Exception:
            logger.exception("rtc input audio loop ended")
        finally:
            if not self._closed:
                await self.close()

    async def barge_in(self) -> None:
        await self.gateway._barge_in()  # noqa: SLF001 - same demo control surface as /ws

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True

            self.output_track.clear()

            current = asyncio.current_task()
            if self._audio_task and self._audio_task is not current:
                self._audio_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._audio_task

            try:
                await self.gateway.stop()
            except Exception:
                logger.exception("gateway session cleanup failed")

            try:
                await self.peer_connection.close()
            except Exception:
                logger.exception("rtc peer cleanup failed")

            self.event_queue.put_nowait(None)
            if self.on_closed is not None:
                self.on_closed()


class RTCGateway:
    def __init__(self) -> None:
        self._calls: dict[str, RTCGatewayCall] = {}

    async def offer(self, business_addr: str, agent_kind: str, sdp: str) -> dict[str, str]:
        session_id = "rtc_" + uuid.uuid4().hex[:10]
        pc = RTCPeerConnection()
        call = RTCGatewayCall(
            session_id=session_id,
            peer_connection=pc,
            business_addr=business_addr,
            agent_kind=agent_kind,
            on_closed=lambda: self._calls.pop(session_id, None),
        )
        self._calls[session_id] = call

        @pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            if track.kind == "audio":
                call.attach_input_track(track)

        @pc.on("connectionstatechange")
        async def on_connection_state_change() -> None:
            if pc.connectionState in ("failed", "closed"):
                await self.close(session_id)

        pc.addTrack(call.output_track)

        try:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await call.start()
        except Exception:
            await call.close()
            raise

        return {"sdp": pc.localDescription.sdp, "session_id": session_id}

    def get(self, session_id: str) -> RTCGatewayCall | None:
        return self._calls.get(session_id)

    async def close(self, session_id: str) -> None:
        call = self._calls.pop(session_id, None)
        if call is not None:
            await call.close()

    async def close_all(self) -> None:
        for session_id in list(self._calls):
            await self.close(session_id)


async def rtc_offer(request: Request) -> JSONResponse:
    body = await request.json()
    sdp = str(body.get("sdp") or "")
    agent_kind = str(body.get("agent") or "cafe_single")
    answer = await request.app.state.rtc_gateway.offer(
        business_addr=request.app.state.business_addr,
        agent_kind=agent_kind,
        sdp=sdp,
    )
    return JSONResponse(answer)


async def rtc_barge_in(request: Request) -> JSONResponse:
    body = await request.json()
    session_id = str(body.get("session_id") or "")
    call = request.app.state.rtc_gateway.get(session_id)
    if call is not None:
        await call.barge_in()
    return JSONResponse({"ok": True})


async def rtc_disconnect(request: Request) -> JSONResponse:
    body = await request.json()
    session_id = str(body.get("session_id") or "")
    await request.app.state.rtc_gateway.close(session_id)
    return JSONResponse({"ok": True})


async def rtc_events(request: Request) -> StreamingResponse:
    session_id = request.path_params["session_id"]
    call = request.app.state.rtc_gateway.get(session_id)

    async def missing() -> AsyncIterator[str]:
        yield 'data: {"kind":"error","message":"session not found"}\n\n'

    if call is None:
        return StreamingResponse(missing(), media_type="text/event-stream")

    async def stream() -> AsyncIterator[str]:
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(call.event_queue.get(), timeout=30)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if event is None:
                    yield 'data: {"kind":"session_ended"}\n\n'
                    break
                import json
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


__all__ = [
    "RTCGateway",
    "rtc_offer",
    "rtc_events",
    "rtc_barge_in",
    "rtc_disconnect",
]
