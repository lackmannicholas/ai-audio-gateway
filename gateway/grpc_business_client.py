"""The gateway's client to the business plane: where proxies are born.

This is the conceptual heart of the architecture. The gateway opens the
bidirectional Bridge stream to the business plane and then:

  * Sends ``call.started`` (with which agent to use) and waits for
    ``session.configure`` carrying the prompt + tool specs.
  * For each ToolSpec, builds a *hollow proxy*: an object the realtime model can
    call, whose entire behavior is "relay this call across the wire and await
    the result." No business logic. No data access. Nothing but a relay.
  * Correlates each in-flight tool call by ``tool_call_id`` using a future, so
    the single multiplexed stream can carry many concurrent tool calls.

Strip the business plane away and these proxies relay to nothing. That is the
point: the gateway holds shapes, not behavior.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from contextlib import suppress
from typing import Any

import grpc

from proto_contract.channel import open_bridge_stream
from proto_contract.envelopes import (
    GatewayCommand,
    GatewayCommandType,
    GatewayEvent,
    GatewayEventType,
    ToolSpec,
)

logger = logging.getLogger("gateway.bridge")


@dataclass
class ProxyTool:
    """A hollow tool the realtime model can call. Relays; never executes.

    Holds the schema (so the realtime model knows how to call it) and a relay
    function (so a call goes across the wire). It has no ``invoke`` of its own
    beyond ``relay``.
    """

    spec: ToolSpec
    _relay: Callable[[str, dict], Awaitable[Any]]

    async def call(self, arguments: dict[str, Any]) -> Any:
        # The whole body of a proxy tool: hand off across the wire.
        return await self._relay(self.spec.name, arguments)


@dataclass
class SessionConfig:
    agent_name: str
    instructions: str
    greeting_instructions: str
    proxies: dict[str, ProxyTool] = field(default_factory=dict)


class BusinessBridgeClient:
    """Gateway-side handle on the business plane over one bidi gRPC stream."""

    def __init__(self, channel: grpc.aio.Channel, metadata: list[tuple[str, str]]) -> None:
        self._channel = channel
        self._metadata = metadata
        self._call: grpc.aio.StreamStreamCall | None = None
        self._reader_task: asyncio.Task | None = None

        # tool_call_id -> future awaiting its ToolCallOutput
        self._pending: dict[str, asyncio.Future] = {}
        # commands that aren't tool outputs (session.configure, etc.)
        self._inbox: asyncio.Queue[GatewayCommand] = asyncio.Queue()
        # informational local-tool-call notifications for the UI
        self.on_local_tool_call: Callable[[str], None] | None = None

        self._call_id = "call_" + uuid.uuid4().hex[:8]
        self._turn_id = 0

    @property
    def call_id(self) -> str:
        return self._call_id

    # -- stream lifecycle ---------------------------------------------------- #
    async def open(self) -> None:
        self._call = open_bridge_stream(self._channel, self._metadata)
        self._reader_task = asyncio.create_task(self._read_loop())

    async def _send_event(self, event: GatewayEvent) -> None:
        assert self._call is not None
        await self._call.write(event.to_json_bytes())

    async def _read_loop(self) -> None:
        assert self._call is not None
        try:
            async for raw in self._call:
                cmd = GatewayCommand.from_json_bytes(raw)
                if cmd.type is GatewayCommandType.TOOL_CALL_OUTPUT:
                    tcid = str(cmd.payload.get("tool_call_id") or "")
                    fut = self._pending.pop(tcid, None)
                    if fut and not fut.done():
                        if cmd.payload.get("is_error"):
                            fut.set_exception(RuntimeError(cmd.payload.get("output_json")))
                        else:
                            fut.set_result(json.loads(cmd.payload.get("output_json") or "null"))
                elif cmd.type is GatewayCommandType.SEND_MESSAGE and \
                        cmd.payload.get("kind") == "local_tool_call":
                    if self.on_local_tool_call:
                        self.on_local_tool_call(str(cmd.payload.get("name")))
                else:
                    await self._inbox.put(cmd)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("bridge read loop ended")

    # -- session setup ------------------------------------------------------- #
    async def start_session(self, agent_kind: str) -> SessionConfig:
        """Send call.started, await session.configure, build the proxies."""
        await self._send_event(GatewayEvent(
            type=GatewayEventType.CALL_STARTED,
            call_id=self._call_id,
            payload={"agent_kind": agent_kind},
        ))
        cmd = await self._inbox.get()
        if cmd.type is not GatewayCommandType.SESSION_CONFIGURE:
            raise RuntimeError(f"expected session.configure, got {cmd.type}")

        proxies: dict[str, ProxyTool] = {}
        for spec_dict in cmd.payload.get("tools", []):
            spec = ToolSpec.model_validate(spec_dict)
            proxies[spec.name] = ProxyTool(spec=spec, _relay=self._relay_tool_call)

        logger.info("session configured for %s: built %d proxy tool(s): %s",
                    cmd.payload.get("agent_name"), len(proxies), list(proxies))
        return SessionConfig(
            agent_name=str(cmd.payload.get("agent_name")),
            instructions=str(cmd.payload.get("instructions")),
            greeting_instructions=str(cmd.payload.get("greeting_instructions")),
            proxies=proxies,
        )

    # -- the relay: the entire behavior of a proxy tool ---------------------- #
    async def _relay_tool_call(self, name: str, arguments: dict[str, Any]) -> Any:
        tool_call_id = "tc_" + uuid.uuid4().hex[:10]
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[tool_call_id] = fut

        await self._send_event(GatewayEvent(
            type=GatewayEventType.TOOL_CALL_REQUESTED,
            call_id=self._call_id,
            payload={
                "name": name,
                "tool_call_id": tool_call_id,
                "arguments_json": json.dumps(arguments),
                "turn_id": self._turn_id,
            },
        ))
        return await fut

    # -- media-plane events the gateway reports ------------------------------ #
    async def barge_in(self) -> int:
        """Caller interrupted: bump turn_id and tell the business plane."""
        self._turn_id += 1
        await self._send_event(GatewayEvent(
            type=GatewayEventType.BARGE_IN,
            call_id=self._call_id,
            payload={"turn_id": self._turn_id},
        ))
        return self._turn_id

    async def end_call(self) -> None:
        try:
            await self._send_event(GatewayEvent(
                type=GatewayEventType.CALL_ENDED, call_id=self._call_id))
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.cancel()
            self._pending.clear()
            if self._call is not None:
                with suppress(Exception):
                    await self._call.done_writing()
            if self._reader_task is not None:
                self._reader_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._reader_task


__all__ = ["BusinessBridgeClient", "ProxyTool", "SessionConfig"]
