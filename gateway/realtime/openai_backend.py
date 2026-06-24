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
import os
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

    async def _connect(self) -> None:
        import websockets  # lazy
        key = os.environ["OPENAI_API_KEY"]
        self._ws = await websockets.connect(
            _url_with_model(_configured_realtime_url(), self._model),
            additional_headers={
                "Authorization": f"Bearer {key}",
                "OpenAI-Safety-Identifier": os.getenv(
                    "OPENAI_SAFETY_IDENTIFIER",
                    "audio-gateway-poc",
                ),
            },
        )
        self._reader = asyncio.create_task(self._read_loop())

    async def configure(self, config: RealtimeSessionConfig) -> None:
        if self._ws is None:
            await self._connect()
        tools = [{"type": "function", "name": t.name, "description": t.description,
                  "parameters": t.params_json_schema} for t in config.tools]
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
                        "transcription": {
                            "model": os.getenv(
                                "TRANSCRIPT_MODEL",
                                "gpt-4o-mini-transcribe",
                            ),
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
        await self._send({"type": "response.cancel"})

    async def submit_tool_output(self, tool_call_id: str, output_json: str) -> None:
        await self._send({
            "type": "conversation.item.create",
            "item": {"type": "function_call_output",
                     "call_id": tool_call_id, "output": output_json},
        })
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
                    await self._out.put(RealtimeEvent(
                        RealtimeEventType.AUDIO_DELTA, {"audio_b64": ev.get("delta")}))
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
                    await self._out.put(RealtimeEvent(RealtimeEventType.RESPONSE_DONE, {}))
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
        return OpenAIRealtimeBackend(os.getenv("REALTIME_MODEL", "gpt-realtime-2"))
    from gateway.realtime.mock_backend import MockRealtimeBackend
    return MockRealtimeBackend()


__all__ = ["OpenAIRealtimeBackend", "build_realtime_backend"]
