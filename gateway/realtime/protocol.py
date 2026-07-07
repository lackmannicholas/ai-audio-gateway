"""The realtime-model interface: the seam between the gateway and whatever
speech-to-speech model is behind it.

The gateway talks to the realtime model only through this protocol, so the model
is swappable: a zero-dependency mock by default, the real OpenAI Realtime API
when ``REALTIME_BACKEND=openai``.

The event vocabulary here is intentionally a small, faithful subset of the
OpenAI Realtime API's server events — enough to drive a full turn:

  inbound (gateway -> model):   append_audio, commit_audio, configure_session,
                                create_response, cancel_response, submit_tool_output
  outbound (model -> gateway):  transcript, audio_delta, tool_call,
                                response_done, speech_started

The gateway translates between these and the business-plane contract. The model
never knows the business plane exists; it just calls tools by name, and the
gateway relays those calls across the wire.
"""

from __future__ import annotations

import abc
import enum
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from proto_contract.envelopes import ToolSpec


class RealtimeEventType(str, enum.Enum):
    TRANSCRIPT = "transcript"            # user or assistant text
    AUDIO_DELTA = "audio_delta"          # a chunk of assistant audio
    TOOL_CALL = "tool_call"              # model wants to call a tool
    RESPONSE_DONE = "response_done"      # assistant turn complete
    SPEECH_STARTED = "speech_started"    # model detected user speech (server VAD)
    ERROR = "error"                      # backend session ended with an error


class RealtimeBackendClosed(RuntimeError):
    """Raised when the backend transport closes while the gateway is active."""


@dataclass
class RealtimeEvent:
    type: RealtimeEventType
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RealtimeSessionConfig:
    instructions: str
    greeting_instructions: str
    tools: list[ToolSpec] = field(default_factory=list)
    voice: str = "verse"
    # Who owns endpointing. There must be exactly one authority: when the
    # gateway's local VAD is gating, the backend disables server-side turn
    # detection; when local VAD is unavailable (e.g. TEN VAD has no build for
    # the platform), the backend leaves server VAD on so the buffer still gets
    # committed and a response still gets created.
    server_vad: bool = False


class RealtimeBackend(abc.ABC):
    """A bidirectional speech-to-speech model session."""

    @abc.abstractmethod
    async def configure(self, config: RealtimeSessionConfig) -> None:
        """Set instructions + tool schemas for the session."""

    @abc.abstractmethod
    async def append_audio(self, pcm: bytes) -> None:
        """Feed inbound caller audio."""

    @abc.abstractmethod
    async def commit_audio(self) -> None:
        """Signal end of a caller utterance (commit the input buffer)."""

    @abc.abstractmethod
    async def create_response(self, instructions: str | None = None) -> None:
        """Ask the model to produce a response (optionally with extra instructions)."""

    @abc.abstractmethod
    async def cancel_response(self) -> None:
        """Interrupt the in-flight response (barge-in)."""

    @abc.abstractmethod
    async def submit_tool_output(self, tool_call_id: str, output_json: str,
                                 create_response: bool = True) -> None:
        """Return a tool result. With ``create_response`` the model continues
        the turn; without it the pending function call is merely resolved
        (used for stale results that must not wake the model back up)."""

    @abc.abstractmethod
    def events(self) -> AsyncIterator[RealtimeEvent]:
        """Async iterator of outbound model events."""

    async def close(self) -> None:  # optional override
        return None


__all__ = [
    "RealtimeBackend",
    "RealtimeBackendClosed",
    "RealtimeEvent",
    "RealtimeEventType",
    "RealtimeSessionConfig",
]
