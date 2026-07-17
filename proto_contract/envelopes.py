"""The wire contract between the audio gateway and the business plane.

This module is the single source of truth for everything that crosses the
gRPC boundary. There is deliberately no ``.proto`` file: the transport is a
gRPC ``stream_stream`` with identity (passthrough) serializers, and the
payloads are these Pydantic models serialized as JSON via ``orjson``.

Why JSON envelopes over protobuf:
  * Human-readable on the wire — you can log a frame and read it.
  * Schema-validated at the edges — Pydantic rejects malformed frames.
  * Still gets gRPC's HTTP/2 multiplexing and bidirectional streaming.

The two enums below ARE the contract. Read them top to bottom and you know
the entire vocabulary of the boundary:

  * ``GatewayEventType``  — things the gateway (media plane) tells the business
    plane *happened*. Past tense. Reports of reality.
  * ``GatewayCommandType`` — things the business plane tells the gateway to
    *do*. Imperative. Requests against the media plane.

The asymmetry is the architecture: the media plane owns reality (audio, turns,
barge-in) and reports it; the business plane owns meaning (agents, tools,
instructions) and directs the media plane through a small command vocabulary.
"""

from __future__ import annotations

import enum
import time
import uuid
from typing import Any

import orjson
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# The vocabulary of the boundary.
# --------------------------------------------------------------------------- #
class GatewayEventType(str, enum.Enum):
    """Gateway -> Business. Things that *happened* in the media plane."""

    # Call lifecycle
    CALL_STARTED = "call.started"
    CALL_ENDED = "call.ended"

    # User speech (driven by VAD)
    USER_SPEECH_STARTED = "user.speech_started"
    USER_SPEECH_STOPPED = "user.speech_stopped"
    # Transcripts, pushed across the wire so the business plane can guardrail
    # them. The user side finalizes once; the assistant side streams as deltas
    # (keyed by response_id) so a guardrail can cancel mid-sentence, before the
    # banned content is fully spoken.
    USER_TRANSCRIPT_COMPLETED = "user.transcript_completed"
    RESPONSE_TRANSCRIPT_DELTA = "response.transcript_delta"

    # Model response lifecycle
    RESPONSE_STARTED = "response.started"
    RESPONSE_DONE = "response.done"

    # The proxy relay: the realtime model invoked a (proxy) tool.
    TOOL_CALL_REQUESTED = "tool_call.requested"

    # Playback / interruption
    PLAYBACK_COMPLETED = "playback.completed"
    BARGE_IN = "barge_in"

    ERROR = "error"


class GatewayCommandType(str, enum.Enum):
    """Business -> Gateway. Things the business plane asks the media plane to do."""

    # Sent once at call start: prompts + tool specs to construct the voice agent.
    SESSION_CONFIGURE = "session.configure"

    # Drive the realtime model.
    RESPONSE_CREATE = "response.create"
    RESPONSE_CANCEL = "response.cancel"

    # The proxy relay return path: result of a tool the business plane executed.
    TOOL_CALL_OUTPUT = "tool_call.output"

    # Inject speech / swap instructions mid-session.
    SEND_MESSAGE = "send_message"
    SESSION_UPDATE_INSTRUCTIONS = "session.update_instructions"

    # The guardrail / interruption control surface (directives against a turn).
    INTERRUPT_POLICY_SET = "interrupt_policy.set"

    ERROR = "error"


# --------------------------------------------------------------------------- #
# Tool specification — what the business plane hands the gateway at configure
# time so the gateway can build hollow proxy tools the realtime model can call.
# --------------------------------------------------------------------------- #
class ToolSpec(BaseModel):
    """The schema of a tool, with no implementation attached.

    The gateway receives a list of these in ``session.configure`` and builds a
    proxy for each. The proxy presents this schema to the realtime model, but
    its invocation does nothing except relay across the wire. Execution lives
    in the business plane.
    """

    name: str
    description: str
    params_json_schema: dict[str, Any] = Field(default_factory=dict)
    strict_json_schema: bool = True


# --------------------------------------------------------------------------- #
# The envelopes.
# --------------------------------------------------------------------------- #
def _now() -> float:
    return time.time()


def _frame_id() -> str:
    return uuid.uuid4().hex


class GatewayEvent(BaseModel):
    """A single Gateway -> Business frame."""

    type: GatewayEventType
    call_id: str
    frame_id: str = Field(default_factory=_frame_id)
    ts: float = Field(default_factory=_now)
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_json_bytes(self) -> bytes:
        return orjson.dumps(self.model_dump(mode="json"))

    @classmethod
    def from_json_bytes(cls, raw: bytes) -> "GatewayEvent":
        return cls.model_validate(orjson.loads(raw))


class GatewayCommand(BaseModel):
    """A single Business -> Gateway frame."""

    type: GatewayCommandType
    call_id: str
    frame_id: str = Field(default_factory=_frame_id)
    ts: float = Field(default_factory=_now)
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_json_bytes(self) -> bytes:
        return orjson.dumps(self.model_dump(mode="json"))

    @classmethod
    def from_json_bytes(cls, raw: bytes) -> "GatewayCommand":
        return cls.model_validate(orjson.loads(raw))


# --------------------------------------------------------------------------- #
# Typed payload helpers. These are not required — payloads are free-form dicts
# on the wire — but they document the expected shape of the important frames
# and give the rest of the codebase something to construct against.
# --------------------------------------------------------------------------- #
class SessionConfigurePayload(BaseModel):
    """Body of a ``session.configure`` command."""

    agent_name: str
    instructions: str
    greeting_instructions: str = "Greet the caller, then wait."
    model: str = "mock-realtime"
    voice: str = "verse"
    tools: list[ToolSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallRequestedPayload(BaseModel):
    """Body of a ``tool_call.requested`` event (gateway proxy -> business)."""

    name: str
    tool_call_id: str
    arguments_json: str = "{}"
    # The turn this tool call belongs to. The business plane snapshots this and
    # compares after slow work to detect staleness (see the thinker).
    turn_id: int = 0


class ToolCallOutputPayload(BaseModel):
    """Body of a ``tool_call.output`` command (business -> gateway proxy)."""

    tool_call_id: str
    output_json: str = "null"
    is_error: bool = False


class BargeInPayload(BaseModel):
    """Body of a ``barge_in`` event. Carries the new (post-increment) turn_id."""

    turn_id: int


class TranscriptPayload(BaseModel):
    """Body of a ``user.transcript_completed`` event, fed to guardrails."""

    role: str  # "user" | "assistant"
    text: str = ""


class TranscriptDeltaPayload(BaseModel):
    """Body of a ``response.transcript_delta`` event.

    An incremental chunk of the assistant's transcript. ``response_id`` groups
    deltas belonging to the same response so the business plane knows when to
    start a fresh transcript buffer.
    """

    role: str = "assistant"
    delta: str = ""
    response_id: str = ""


class ResponseCancelPayload(BaseModel):
    """Body of a ``response.cancel`` command (business -> gateway).

    Sent when a guardrail trips: the gateway cancels the in-flight response and
    clears the audio queues. ``rule``/``reason`` are informational (for the UI
    and logs) — the gateway acts on the command regardless of them.
    """

    rule: str | None = None
    reason: str | None = None


__all__ = [
    "GatewayEventType",
    "GatewayCommandType",
    "ToolSpec",
    "GatewayEvent",
    "GatewayCommand",
    "SessionConfigurePayload",
    "ToolCallRequestedPayload",
    "ToolCallOutputPayload",
    "BargeInPayload",
    "TranscriptPayload",
    "TranscriptDeltaPayload",
    "ResponseCancelPayload",
]
