"""Endpointing must have exactly one authority.

When local VAD gates (TEN VAD present), the gateway tells the realtime backend
to disable server-side turn detection. When local VAD is the passthrough
fallback (e.g. TEN VAD has no build for the platform — Linux aarch64 / Docker
on Apple Silicon), the gateway must keep server VAD on, or nothing commits the
audio buffer and no response is ever created.

This is a regression guard for exactly that: disabling server turn detection
unconditionally silently breaks every platform without a local VAD build.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import grpc
import pytest

from business.grpc_server import BusinessBridgeServer
from gateway.audio.vad import PassthroughVAD, VADResult
from gateway.realtime.openai_backend import _turn_detection_for
from gateway.realtime.protocol import (
    RealtimeBackend,
    RealtimeEvent,
    RealtimeSessionConfig,
)
from gateway.session import GatewaySession


class ConfigCapturingBackend(RealtimeBackend):
    """Records the RealtimeSessionConfig it is configured with."""

    def __init__(self) -> None:
        self.config: RealtimeSessionConfig | None = None

    async def configure(self, config: RealtimeSessionConfig) -> None:
        self.config = config

    async def append_audio(self, pcm: bytes) -> None:
        return None

    async def commit_audio(self) -> None:
        return None

    async def create_response(self, instructions: str | None = None) -> None:
        return None

    async def cancel_response(self) -> None:
        return None

    async def submit_tool_output(self, tool_call_id: str, output_json: str,
                                 create_response: bool = True) -> None:
        return None

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        while False:
            yield  # pragma: no cover - never emits


class GatingVAD:
    """A local VAD that actually gates (not the passthrough fallback)."""

    def process(self, pcm16: bytes) -> VADResult:
        return VADResult(is_speech=False, speech_probability=0.0,
                         frames_to_flush=[pcm16] if pcm16 else [])


async def _capture_config(vad, bind: str) -> RealtimeSessionConfig:
    server = BusinessBridgeServer()
    await server.start(bind=bind)
    backend = ConfigCapturingBackend()
    try:
        session = GatewaySession(
            realtime=backend,
            business_addr=bind,
            agent_kind="cafe_single",
            ui=lambda kind, data: None,
            vad=vad,
        )
        await session.start()
        await session.stop()
    finally:
        await server.stop()
    assert backend.config is not None
    return backend.config


@pytest.mark.asyncio
async def test_passthrough_vad_keeps_server_vad_on(monkeypatch):
    monkeypatch.setenv("BRIDGE_INSECURE", "1")
    config = await _capture_config(PassthroughVAD(), "127.0.0.1:8211")
    assert config.server_vad is True


@pytest.mark.asyncio
async def test_gating_vad_disables_server_vad(monkeypatch):
    monkeypatch.setenv("BRIDGE_INSECURE", "1")
    config = await _capture_config(GatingVAD(), "127.0.0.1:8212")
    assert config.server_vad is False


def test_backend_maps_server_vad_flag_to_turn_detection():
    """server_vad True -> a server_vad turn_detection block; False -> null."""
    assert _turn_detection_for(True) == {"type": "server_vad"}
    assert _turn_detection_for(False) is None
