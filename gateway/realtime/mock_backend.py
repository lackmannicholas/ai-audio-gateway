"""The mock realtime backend: a zero-dependency stand-in for OpenAI Realtime.

It is NOT an LLM. It is a scripted state machine that speaks the same event
protocol a real realtime model would, so the entire system runs end-to-end with
no API key and no cost. It exists to exercise the architecture — the proxy
relay, tool calls crossing the wire, barge-in, turn staleness — deterministically.

When a caller utterance is committed (or ``create_response`` is called), the
mock:
  1. emits a transcript of what it "heard",
  2. decides whether the turn needs a tool, and if so emits a ``tool_call``
     (which the gateway relays across the wire to the business plane),
  3. waits for the tool output via ``submit_tool_output``,
  4. emits assistant ``audio_delta`` chunks + a transcript,
  5. emits ``response_done``.

Barge-in (``cancel_response``) interrupts step 4 mid-stream, exactly as a real
model would when the caller talks over it.

Audio is faked as silence frames sized to a target duration — enough for the
playback/pacing path and barge-in timing to be real, without needing TTS.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from gateway.realtime.protocol import (
    RealtimeBackend,
    RealtimeEvent,
    RealtimeEventType,
    RealtimeSessionConfig,
)

# 20ms of 24 kHz 16-bit mono PCM = 960 bytes. Matches the gateway's audio path.
_FRAME_BYTES = 960
_FRAMES_PER_RESPONSE = 25  # ~500ms of audio per assistant turn


class MockRealtimeBackend(RealtimeBackend):
    def __init__(self, scripted_utterance: str = "I'll have a large oat latte") -> None:
        self._config: RealtimeSessionConfig | None = None
        self._out: asyncio.Queue[RealtimeEvent] = asyncio.Queue()
        self._scripted = scripted_utterance
        self._pending_tool: asyncio.Future | None = None
        self._turn_task: asyncio.Task | None = None  # keep a ref: bare create_task can be GC'd
        self._cancelled = False
        self._tool_names: set[str] = set()

    async def configure(self, config: RealtimeSessionConfig) -> None:
        self._config = config
        self._tool_names = {t.name for t in config.tools}

    async def append_audio(self, pcm: bytes) -> None:
        # The mock ignores audio content; real backend would buffer/transcribe.
        return None

    async def commit_audio(self) -> None:
        # A committed utterance triggers a response.
        await self.create_response()

    async def create_response(self, instructions: str | None = None) -> None:
        self._cancelled = False
        self._turn_task = asyncio.create_task(self._run_turn())

    async def cancel_response(self) -> None:
        self._cancelled = True
        if self._pending_tool and not self._pending_tool.done():
            self._pending_tool.cancel()

    async def submit_tool_output(self, tool_call_id: str, output_json: str,
                                 create_response: bool = True) -> None:
        if self._pending_tool and not self._pending_tool.done():
            self._pending_tool.set_result(output_json)

    async def _run_turn(self) -> None:
        # 1. transcript of what we "heard"
        await self._out.put(RealtimeEvent(
            RealtimeEventType.TRANSCRIPT,
            {"role": "user", "text": self._scripted}))

        # 2. decide on a tool. The mock prefers consult_thinker if present
        #    (responder/thinker agent), else get_menu+place_order (single agent).
        if self._cancelled:
            return
        if "consult_thinker" in self._tool_names:
            out = await self._call_tool("consult_thinker",
                                        {"request": self._scripted})
            spoken = self._spoken_from_tool(out)
        elif "place_order" in self._tool_names:
            await self._call_tool("get_menu", {})
            if self._cancelled:
                return
            out = await self._call_tool(
                "place_order",
                {"items": [{"drink": "latte", "size": "large", "milk": "oat"}]})
            spoken = self._spoken_from_tool(out)
        else:
            spoken = "Sorry, I don't have any tools configured."

        if self._cancelled:
            return

        # 3. assistant transcript + audio
        await self._out.put(RealtimeEvent(
            RealtimeEventType.TRANSCRIPT,
            {"role": "assistant", "text": spoken}))
        for _ in range(_FRAMES_PER_RESPONSE):
            if self._cancelled:
                return
            await self._out.put(RealtimeEvent(
                RealtimeEventType.AUDIO_DELTA,
                {"pcm_len": _FRAME_BYTES}))
            await asyncio.sleep(0.02)  # real 20ms cadence so barge-in timing is real

        if not self._cancelled:
            await self._out.put(RealtimeEvent(RealtimeEventType.RESPONSE_DONE, {}))

    async def _call_tool(self, name: str, args: dict) -> str:
        """Emit a tool_call event and await the gateway's submit_tool_output."""
        self._pending_tool = asyncio.get_running_loop().create_future()
        tool_call_id = f"mock_{name}"
        await self._out.put(RealtimeEvent(
            RealtimeEventType.TOOL_CALL,
            {"name": name, "tool_call_id": tool_call_id,
             "arguments_json": json.dumps(args)}))
        try:
            return await self._pending_tool
        except asyncio.CancelledError:
            return "null"

    @staticmethod
    def _spoken_from_tool(output_json: str) -> str:
        try:
            data = json.loads(output_json)
        except (json.JSONDecodeError, TypeError):
            return "All set."
        if isinstance(data, dict):
            if "answer" in data:                 # thinker result
                return str(data["answer"])
            if data.get("ok") and "total" in data:  # place_order result
                return f"Order in — that's ${data['total']:.2f}."
        return "All set — anything else?"

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        while True:
            yield await self._out.get()


__all__ = ["MockRealtimeBackend"]
