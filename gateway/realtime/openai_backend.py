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

from gateway.realtime.protocol import (
    RealtimeBackend,
    RealtimeEvent,
    RealtimeEventType,
    RealtimeSessionConfig,
)

_REALTIME_URL = "wss://api.openai.com/v1/realtime"


class OpenAIRealtimeBackend(RealtimeBackend):
    def __init__(self, model: str = "gpt-realtime") -> None:
        self._model = model
        self._ws = None
        self._out: asyncio.Queue[RealtimeEvent] = asyncio.Queue()
        self._reader: asyncio.Task | None = None

    async def _connect(self) -> None:
        import websockets  # lazy
        key = os.environ["OPENAI_API_KEY"]
        self._ws = await websockets.connect(
            f"{_REALTIME_URL}?model={self._model}",
            additional_headers={"Authorization": f"Bearer {key}",
                                "OpenAI-Beta": "realtime=v1"},
        )
        self._reader = asyncio.create_task(self._read_loop())

    async def configure(self, config: RealtimeSessionConfig) -> None:
        if self._ws is None:
            await self._connect()
        tools = [{"type": "function", "name": t.name, "description": t.description,
                  "parameters": t.params_json_schema} for t in config.tools]
        await self._send({
            "type": "session.update",
            "session": {"instructions": config.instructions, "voice": config.voice,
                        "tools": tools, "tool_choice": "auto",
                        "input_audio_format": "g711_ulaw",
                        "output_audio_format": "g711_ulaw"},
        })

    async def append_audio(self, pcm: bytes) -> None:
        import base64
        await self._send({"type": "input_audio_buffer.append",
                          "audio": base64.b64encode(pcm).decode()})

    async def commit_audio(self) -> None:
        await self._send({"type": "input_audio_buffer.commit"})

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
        assert self._ws is not None
        await self._ws.send(json.dumps(msg))

    async def _read_loop(self) -> None:
        assert self._ws is not None
        async for raw in self._ws:
            ev = json.loads(raw)
            t = ev.get("type", "")
            if t == "response.audio.delta":
                await self._out.put(RealtimeEvent(
                    RealtimeEventType.AUDIO_DELTA, {"audio_b64": ev.get("delta")}))
            elif t in ("response.audio_transcript.delta",
                       "conversation.item.input_audio_transcription.completed"):
                await self._out.put(RealtimeEvent(
                    RealtimeEventType.TRANSCRIPT,
                    {"role": "assistant" if "audio_transcript" in t else "user",
                     "text": ev.get("delta") or ev.get("transcript", "")}))
            elif t == "response.function_call_arguments.done":
                await self._out.put(RealtimeEvent(
                    RealtimeEventType.TOOL_CALL,
                    {"name": ev.get("name"), "tool_call_id": ev.get("call_id"),
                     "arguments_json": ev.get("arguments", "{}")}))
            elif t == "input_audio_buffer.speech_started":
                await self._out.put(RealtimeEvent(RealtimeEventType.SPEECH_STARTED, {}))
            elif t == "response.done":
                await self._out.put(RealtimeEvent(RealtimeEventType.RESPONSE_DONE, {}))

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        while True:
            yield await self._out.get()

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
        if self._ws is not None:
            await self._ws.close()


def build_realtime_backend() -> RealtimeBackend:
    """Factory honoring REALTIME_BACKEND (mock | openai)."""
    if os.getenv("REALTIME_BACKEND", "mock").lower() == "openai":
        return OpenAIRealtimeBackend(os.getenv("REALTIME_MODEL", "gpt-realtime"))
    from gateway.realtime.mock_backend import MockRealtimeBackend
    return MockRealtimeBackend()


__all__ = ["OpenAIRealtimeBackend", "build_realtime_backend"]
