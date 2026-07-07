"""Barge-in must fire during the playback tail, not just while generating.

The realtime model streams audio faster than realtime, so a response finishes
generating — response_active flips false — while seconds of it are still queued
for paced playback. A caller who interrupts during that tail must still trigger
a barge-in (which clears the queued audio), otherwise the old audio keeps
playing over the interrupted conversation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from gateway.audio.vad import VADResult
from gateway.realtime.protocol import (
    RealtimeBackend,
    RealtimeEvent,
    RealtimeSessionConfig,
)
from gateway.session import GatewaySession


class RecordingBackend(RealtimeBackend):
    def __init__(self) -> None:
        self.cancelled = 0
        self.appended: list[bytes] = []

    async def configure(self, config: RealtimeSessionConfig) -> None:
        return None

    async def append_audio(self, pcm: bytes) -> None:
        self.appended.append(pcm)

    async def commit_audio(self) -> None:
        return None

    async def create_response(self, instructions: str | None = None) -> None:
        return None

    async def cancel_response(self) -> None:
        self.cancelled += 1

    async def submit_tool_output(self, tool_call_id: str, output_json: str,
                                 create_response: bool = True) -> None:
        return None

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        while False:
            yield  # pragma: no cover


class SpeechOnsetVAD:
    """Reports speech onset on the first chunk it sees."""

    def process(self, pcm16: bytes) -> VADResult:
        return VADResult(is_speech=True, speech_probability=1.0,
                         frames_to_flush=[pcm16], speech_started=True)


def _session(pending: bool) -> tuple[GatewaySession, RecordingBackend, list]:
    backend = RecordingBackend()
    ui: list[tuple[str, dict]] = []
    session = GatewaySession(
        realtime=backend,
        ui=lambda kind, data: ui.append((kind, data)),
        vad=SpeechOnsetVAD(),
        pending_audio=lambda: pending,
    )
    return session, backend, ui


@pytest.mark.asyncio
async def test_barge_in_fires_when_audio_still_queued():
    session, backend, ui = _session(pending=True)
    # response_active is False (the response finished generating) but audio is
    # still queued — the caller interrupts during the tail.
    assert session._interrupt.response_active is False
    await session.on_caller_audio(b"\x00\x00" * 240)
    assert backend.cancelled == 1
    assert any(kind == "barge_in" for kind, _ in ui)


@pytest.mark.asyncio
async def test_no_barge_in_when_idle_and_no_queued_audio():
    session, backend, ui = _session(pending=False)
    assert session._interrupt.response_active is False
    await session.on_caller_audio(b"\x00\x00" * 240)
    assert backend.cancelled == 0
    assert not any(kind == "barge_in" for kind, _ in ui)
