"""The real OpenAI Realtime backend adapter. Used when REALTIME_BACKEND=openai.

This adapts the OpenAI Realtime API's websocket event stream to our internal
``RealtimeBackend`` protocol. It is deliberately the *only* file that knows
OpenAI's wire format — everything else in the gateway speaks our internal
``RealtimeEvent`` vocabulary, so swapping providers means writing one more
adapter, not touching the gateway.

Kept import-lazy and untested-by-default so the repo runs with zero external
deps in mock mode. Real usage needs OPENAI_API_KEY and the ``websockets``
package. The mapping below is faithful to the Realtime API event names as of
this writing; treat it as a reference adapter, not a maintained client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from urllib.parse import urlparse

from gateway.realtime.protocol import (
    RealtimeBackend,
    RealtimeBackendClosed,
    RealtimeEvent,
    RealtimeEventType,
    RealtimeSessionConfig,
)

logger = logging.getLogger("gateway.realtime.openai")

_DEFAULT_REALTIME_URL = "wss://us.api.openai.com/v1/realtime"


def _base_url_to_realtime_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url[len("https://"):]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://"):]
    return f"{url}/realtime"


def _configured_realtime_url() -> str:
    explicit_url = os.getenv("OPENAI_REALTIME_URL") or os.getenv("REALTIME_API_URL")
    if explicit_url:
        return explicit_url
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return _base_url_to_realtime_url(base_url)
    return _DEFAULT_REALTIME_URL


def _url_with_model(url: str, model: str) -> str:
    parsed = urlparse(url)
    if "model=" in parsed.query:
        return url
    sep = "&" if parsed.query else "?"
    return f"{url}{sep}model={model}"


def _turn_detection_for(server_vad: bool) -> dict | None:
    """The ``turn_detection`` block for a session. Server VAD when the gateway's
    local VAD isn't gating, otherwise ``None`` (local VAD owns endpointing)."""
    return {"type": "server_vad"} if server_vad else None


def _describe_ws_close(exc: Exception) -> str:
    received = getattr(exc, "rcvd", None)
    code = getattr(received, "code", None)
    reason = getattr(received, "reason", "") or str(exc)
    details = f"{code}: {reason}" if code is not None else reason
    message = f"OpenAI Realtime websocket closed ({details})"
    if "incorrect_hostname" in reason:
        message += (
            "; check OPENAI_BASE_URL or set "
            "OPENAI_REALTIME_URL=wss://us.api.openai.com/v1/realtime"
        )
    return message


class OpenAIRealtimeBackend(RealtimeBackend):
    def __init__(self, model: str = "gpt-realtime-2") -> None:
        self._model = model
        self._ws = None
        self._out: asyncio.Queue[RealtimeEvent] = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        # Barge-in bookkeeping. A response.cancel is asynchronous: audio deltas
        # for the cancelled response keep arriving until the server processes
        # it, so we track the active response id and drop deltas from cancelled
        # responses instead of letting them re-fill the playback queue.
        self._active_response_id: str | None = None
        self._active_item_id: str | None = None
        self._audio_started_at: float | None = None
        self._cancelled_response_ids: set[str] = set()
        self._server_vad = False

    async def _connect(self) -> None:
        import websockets  # lazy
        key = os.environ["OPENAI_API_KEY"]
        self._ws = await websockets.connect(
            _url_with_model(_configured_realtime_url(), self._model),
            additional_headers={
                "Authorization": f"Bearer {key}",
                "OpenAI-Safety-Identifier": os.getenv(
                    "OPENAI_SAFETY_IDENTIFIER") or "audio-gateway-poc",
            },
        )
        self._reader = asyncio.create_task(self._read_loop())

    async def configure(self, config: RealtimeSessionConfig) -> None:
        if self._ws is None:
            await self._connect()
        tools = [{"type": "function", "name": t.name, "description": t.description,
                  "parameters": t.params_json_schema} for t in config.tools]
        # Exactly one endpointing authority. With local VAD gating we disable
        # server turn detection (otherwise both VADs fire and every utterance
        # triggers duplicate responses); without it we keep server VAD on so
        # the buffer still gets committed and a response still gets created.
        self._server_vad = config.server_vad
        turn_detection = _turn_detection_for(config.server_vad)
        await self._send({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["audio"],
                "instructions": config.instructions,
                "tools": tools,
                "tool_choice": "auto",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "turn_detection": turn_detection,
                        "transcription": {
                            "model": os.getenv(
                                "TRANSCRIPT_MODEL") or "gpt-4o-mini-transcribe",
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": config.voice,
                    },
                },
            },
        })

    async def append_audio(self, pcm: bytes) -> None:
        import base64
        await self._send({"type": "input_audio_buffer.append",
                          "audio": base64.b64encode(pcm).decode()})

    async def commit_audio(self) -> None:
        await self._send({"type": "input_audio_buffer.commit"})
        await self._send({"type": "response.create"})

    async def create_response(self, instructions: str | None = None) -> None:
        msg: dict = {"type": "response.create"}
        if instructions:
            msg["response"] = {"instructions": instructions}
        await self._send(msg)

    async def cancel_response(self) -> None:
        # Nothing is being generated — e.g. the demo barge-in button pressed
        # while the assistant is idle. Skip the cancel so the server doesn't
        # reject it with response_cancel_not_active, and skip the truncate
        # (there is no item to truncate).
        if self._active_response_id is None:
            return
        self._cancelled_response_ids.add(self._active_response_id)
        await self._send({"type": "response.cancel"})
        # Truncate the assistant item to what the caller actually heard.
        # Without this the model's context contains the full response text and
        # later turns are grounded in words that were never spoken. Playback is
        # paced in real time, so elapsed-since-first-delta approximates the
        # audio milliseconds actually played.
        if self._active_item_id is not None and self._audio_started_at is not None:
            audio_end_ms = max(0, int((time.monotonic() - self._audio_started_at) * 1000))
            await self._send({
                "type": "conversation.item.truncate",
                "item_id": self._active_item_id,
                "content_index": 0,
                "audio_end_ms": audio_end_ms,
            })
        self._active_response_id = None
        self._active_item_id = None
        self._audio_started_at = None

    async def submit_tool_output(self, tool_call_id: str, output_json: str,
                                 create_response: bool = True) -> None:
        await self._send({
            "type": "conversation.item.create",
            "item": {"type": "function_call_output",
                     "call_id": tool_call_id, "output": output_json},
        })
        if create_response:
            await self._send({"type": "response.create"})

    async def _send(self, msg: dict) -> None:
        import websockets  # lazy

        if self._ws is None:
            raise RealtimeBackendClosed("OpenAI Realtime websocket is not connected")
        try:
            await self._ws.send(json.dumps(msg))
        except websockets.exceptions.ConnectionClosed as exc:
            raise RealtimeBackendClosed(_describe_ws_close(exc)) from exc

    async def _read_loop(self) -> None:
        import websockets  # lazy

        assert self._ws is not None
        try:
            async for raw in self._ws:
                ev = json.loads(raw)
                t = ev.get("type", "")
                if t in ("response.output_audio.delta", "response.audio.delta"):
                    if ev.get("response_id") in self._cancelled_response_ids:
                        continue  # in-flight delta from a barged-in response
                    if self._audio_started_at is None:
                        self._audio_started_at = time.monotonic()
                    self._active_item_id = ev.get("item_id") or self._active_item_id
                    await self._out.put(RealtimeEvent(
                        RealtimeEventType.AUDIO_DELTA, {"audio_b64": ev.get("delta")}))
                elif t == "response.created":
                    self._active_response_id = (ev.get("response") or {}).get("id")
                    self._active_item_id = None
                    self._audio_started_at = None
                elif t == "conversation.item.input_audio_transcription.completed":
                    await self._out.put(RealtimeEvent(
                        RealtimeEventType.TRANSCRIPT,
                        {"role": "user", "text": ev.get("transcript", "")}))
                elif t in ("response.output_audio_transcript.done",
                           "response.audio_transcript.done"):
                    await self._out.put(RealtimeEvent(
                        RealtimeEventType.TRANSCRIPT,
                        {"role": "assistant", "text": ev.get("transcript", "")}))
                elif t == "response.function_call_arguments.done":
                    await self._out.put(RealtimeEvent(
                        RealtimeEventType.TOOL_CALL,
                        {"name": ev.get("name"), "tool_call_id": ev.get("call_id"),
                         "arguments_json": ev.get("arguments", "{}")}))
                elif t == "input_audio_buffer.speech_started":
                    await self._out.put(RealtimeEvent(
                        RealtimeEventType.SPEECH_STARTED, {}))
                elif t == "response.done":
                    response_id = (ev.get("response") or {}).get("id")
                    was_cancelled = response_id in self._cancelled_response_ids
                    self._cancelled_response_ids.discard(response_id)
                    if response_id == self._active_response_id:
                        self._active_response_id = None
                    if not was_cancelled:
                        await self._out.put(RealtimeEvent(RealtimeEventType.RESPONSE_DONE, {}))
                elif t == "error":
                    # Non-fatal API errors (the transport is still up). Surface
                    # them in the gateway log instead of dropping them silently.
                    err = ev.get("error") or {}
                    logger.warning("OpenAI Realtime error: %s (%s)",
                                   err.get("message"), err.get("code"))
        except websockets.exceptions.ConnectionClosed as exc:
            await self._out.put(RealtimeEvent(
                RealtimeEventType.ERROR,
                {"message": _describe_ws_close(exc)},
            ))

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        while True:
            yield await self._out.get()

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader
            self._reader = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


def build_realtime_backend() -> RealtimeBackend:
    """Factory honoring REALTIME_BACKEND (mock | openai)."""
    if os.getenv("REALTIME_BACKEND", "mock").lower() == "openai":
        return OpenAIRealtimeBackend(os.getenv("REALTIME_MODEL") or "gpt-realtime-2")
    from gateway.realtime.mock_backend import MockRealtimeBackend
    return MockRealtimeBackend()


__all__ = ["OpenAIRealtimeBackend", "build_realtime_backend"]
