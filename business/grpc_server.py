"""The business plane's gRPC server: the relay.

It serves the bidirectional Bridge stream. The gateway connects, sends events
(call.started, tool_call.requested, barge_in, ...), and this server responds
with commands (session.configure, tool_call.output, ...).

Responsibilities:
  * On ``call.started``: pick the agent (single or responder/thinker), render
    its instructions, and reply with ``session.configure`` carrying the prompt
    and the tool specs the gateway should build proxies for.
  * On ``tool_call.requested``: execute the named tool in this process (off the
    media hot path) and reply with ``tool_call.output``. For the thinker tool,
    this is where the nested fan-out happens — invisibly to the gateway.
  * On ``barge_in``: record the new turn_id so in-flight thinker work can detect
    it went stale.

The agent is chosen from session metadata so the SAME gateway can drive either
agent — that selection is the only thing that differs between the two.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator

import grpc

from business.agents.base import VoiceAgent
from business.agents.responder_thinker import ResponderThinkerAgent
from business.agents.single_agent import SingleVoiceAgent
from business.guardrails import GuardrailVerdict, build_guardrails
from business.tools.base import ToolContext
from proto_contract.auth import verify_token
from proto_contract.channel import add_bridge_handler
from proto_contract.envelopes import (
    GatewayCommand,
    GatewayCommandType,
    GatewayEvent,
    GatewayEventType,
)

logger = logging.getLogger("business.grpc")


def _select_agent(agent_kind: str) -> VoiceAgent:
    if agent_kind == "cafe_responder_thinker" or agent_kind == "rt":
        return ResponderThinkerAgent()
    return SingleVoiceAgent()


def _response_cancel_cmd(call_id: str, verdict: GuardrailVerdict) -> bytes:
    """A ``response.cancel`` command carrying the guardrail's reason."""
    return GatewayCommand(
        type=GatewayCommandType.RESPONSE_CANCEL,
        call_id=call_id,
        payload={"rule": verdict.rule, "reason": verdict.reason},
    ).to_json_bytes()


class BusinessSession:
    """Per-call state living in the business plane."""

    def __init__(self, call_id: str, agent: VoiceAgent) -> None:
        self.call_id = call_id
        self.agent = agent
        self.toolset = agent.exposed_toolset()
        # Safety policy for this call. Lives here in the business plane, not in
        # the media plane or the model prompt: centralized, testable, swappable.
        self.guardrails = build_guardrails()
        # Streaming guardrail state: the assistant transcript accumulates per
        # response so a blocked topic split across deltas is still caught, and
        # so we cancel a given response at most once.
        self._assistant_transcript = ""
        self._guardrail_response_id: str | None = None
        self._guardrail_tripped = False
        # Mirrors the gateway's turn_id. Updated on barge_in. The thinker
        # snapshots this before slow work and compares after (staleness).
        self.current_turn_id = 0
        # Outbound queue of "local" tool-call notifications from the thinker,
        # surfaced to the gateway as informational events for the UI.
        self.local_tool_events: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()

        # Wire the thinker's nested calls to our local-event queue, if present.
        thinker = getattr(agent, "_thinker", None)
        if thinker is not None:
            async def _note(name: str, args: dict) -> None:
                await self.local_tool_events.put((name, args))
            thinker.on_tool_call = _note

    def check_assistant_delta(self, response_id: str, delta: str) -> GuardrailVerdict:
        """Accumulate a streaming assistant transcript delta and re-evaluate.

        Buffering per ``response_id`` means a topic split across deltas
        ("weat" + "her") is still caught, and a fresh response starts a fresh
        buffer. Once a response is flagged, later deltas for it are ignored so we
        never cancel the same response twice.
        """
        if response_id != self._guardrail_response_id:
            self._guardrail_response_id = response_id
            self._assistant_transcript = ""
            self._guardrail_tripped = False
        if self._guardrail_tripped:
            return GuardrailVerdict(allowed=True)
        self._assistant_transcript += delta
        verdict = self.guardrails.evaluate("assistant", self._assistant_transcript)
        if not verdict.allowed:
            self._guardrail_tripped = True
        return verdict

    def tool_context(self, snapshot_turn: int) -> ToolContext:
        """Build the runtime context for one tool invocation.

        ``is_stale`` closes over this session, so it always compares the
        invocation's snapshot against the *live* turn. A barge-in advances
        current_turn_id, which makes any in-flight tool/thinker run stale.
        """
        return ToolContext(
            call_id=self.call_id,
            turn_id=snapshot_turn,
            is_stale=lambda: snapshot_turn != self.current_turn_id,
        )


class BusinessBridgeServer:
    def __init__(self) -> None:
        self._server: grpc.aio.Server | None = None

    async def _handler(
        self,
        request_iter: AsyncIterator[bytes],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[bytes]:
        # --- auth: token-in-metadata, checked once at stream open ---
        md = dict(context.invocation_metadata() or [])
        if not verify_token(md.get("x-bridge-token")):
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "bad or missing bridge token")
            return

        session: BusinessSession | None = None
        # Outbound command queue: lets tool execution run concurrently with
        # event processing. Without this, awaiting a slow tool call would block
        # the handler from receiving a barge_in — exactly the bug that makes
        # staleness detection impossible. The media plane never blocks; neither
        # should our handling of its events.
        outbox: asyncio.Queue[bytes | None] = asyncio.Queue()
        tool_tasks: set[asyncio.Task] = set()

        async def pump_inbound() -> None:
            nonlocal session
            async for raw in request_iter:
                event = GatewayEvent.from_json_bytes(raw)

                if event.type is GatewayEventType.CALL_STARTED:
                    agent_kind = str(event.payload.get("agent_kind") or "cafe_single")
                    try:
                        agent = _select_agent(agent_kind)
                        session = BusinessSession(event.call_id, agent)
                    except Exception as exc:  # noqa: BLE001 - report, don't hang
                        # e.g. THINKER_BACKEND=openai but the `openai` extra
                        # isn't installed. If we let this propagate, the gateway
                        # waits forever for session.configure. Surface it so the
                        # gateway fails the connect with a clear message instead.
                        logger.exception("agent init failed for %s", agent_kind)
                        await outbox.put(GatewayCommand(
                            type=GatewayCommandType.ERROR,
                            call_id=event.call_id,
                            payload={"message": f"agent init failed: {exc}"},
                        ).to_json_bytes())
                        break
                    logger.info("call %s -> agent %s (%d exposed tools)",
                                event.call_id, agent.name, len(session.toolset.specs()))
                    await outbox.put(GatewayCommand(
                        type=GatewayCommandType.SESSION_CONFIGURE,
                        call_id=event.call_id,
                        payload={
                            "agent_name": agent.name,
                            "instructions": agent.instructions,
                            "greeting_instructions": agent.greeting_instructions,
                            "tools": [s.model_dump() for s in session.toolset.specs()],
                        },
                    ).to_json_bytes())

                elif event.type is GatewayEventType.TOOL_CALL_REQUESTED:
                    if session is None:
                        continue
                    # Run the tool concurrently so we keep receiving events
                    # (notably barge_in) while it executes.
                    task = asyncio.create_task(self._run_tool(session, event, outbox))
                    tool_tasks.add(task)
                    task.add_done_callback(tool_tasks.discard)

                elif event.type is GatewayEventType.BARGE_IN:
                    if session is not None:
                        session.current_turn_id = int(event.payload.get("turn_id") or 0)
                        logger.info("barge_in: turn_id -> %d", session.current_turn_id)

                elif event.type is GatewayEventType.USER_TRANSCRIPT_COMPLETED:
                    # Input guardrail: check the caller's finalized transcript.
                    if session is None:
                        continue
                    verdict = session.guardrails.evaluate(
                        "user", str(event.payload.get("text") or ""))
                    if not verdict.allowed:
                        logger.info("guardrail %s blocked user transcript: %s",
                                    verdict.rule, verdict.reason)
                        await outbox.put(
                            _response_cancel_cmd(event.call_id, verdict))

                elif event.type is GatewayEventType.RESPONSE_TRANSCRIPT_DELTA:
                    # Streaming output guardrail: accumulate the assistant
                    # transcript and re-check on every delta, so a violation is
                    # caught the instant it forms and the response is cancelled
                    # mid-sentence. The policy is here in the business plane; the
                    # enforcement is a single command back to the media plane.
                    if session is None:
                        continue
                    verdict = session.check_assistant_delta(
                        str(event.payload.get("response_id") or ""),
                        str(event.payload.get("delta") or ""))
                    if not verdict.allowed:
                        logger.info("guardrail %s blocked assistant transcript: %s",
                                    verdict.rule, verdict.reason)
                        await outbox.put(
                            _response_cancel_cmd(event.call_id, verdict))

                elif event.type is GatewayEventType.CALL_ENDED:
                    logger.info("call %s ended", event.call_id)
                    break
            await outbox.put(None)  # sentinel: stop the drain

        pump_task = asyncio.create_task(pump_inbound())
        try:
            while True:
                item = await outbox.get()
                if item is None:
                    break
                yield item
        finally:
            pump_task.cancel()
            for t in tool_tasks:
                t.cancel()

    async def _run_tool(self, session: "BusinessSession", event: GatewayEvent,
                        outbox: asyncio.Queue) -> None:
        """Execute one tool call concurrently and enqueue the result.

        Runs as its own task so the inbound event pump keeps flowing — a
        barge_in can arrive and bump turn_id while this is still working, which
        is precisely what lets the thinker detect staleness.
        """
        name = str(event.payload.get("name") or "")
        tool_call_id = str(event.payload.get("tool_call_id") or "")
        snapshot_turn = int(event.payload.get("turn_id") or 0)
        if snapshot_turn > session.current_turn_id:
            session.current_turn_id = snapshot_turn
        ctx = session.tool_context(snapshot_turn)

        args: dict = {}
        args_json = event.payload.get("arguments_json")
        if args_json:
            try:
                args = json.loads(args_json)
            except json.JSONDecodeError:
                args = {}

        # Surface nested local tool-call notifications as they happen (the
        # thinker's fan-out), so the UI sees the tree before the final output.
        async def drain_local() -> None:
            while True:
                lname, _largs = await session.local_tool_events.get()
                await outbox.put(GatewayCommand(
                    type=GatewayCommandType.SEND_MESSAGE,
                    call_id=event.call_id,
                    payload={"kind": "local_tool_call", "name": lname},
                ).to_json_bytes())

        drain_task = asyncio.create_task(drain_local())
        try:
            result = await session.toolset.invoke(name, args, ctx)
            is_error = False
        except Exception as exc:  # noqa: BLE001 - surface to gateway
            result = {"error": str(exc)}
            is_error = True
            logger.exception("tool %s failed", name)
        finally:
            drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain_task
            # Flush notifications the drain task hadn't picked up yet, so the
            # UI's local-call tree can't lose its trailing nodes.
            while not session.local_tool_events.empty():
                lname, _largs = session.local_tool_events.get_nowait()
                await outbox.put(GatewayCommand(
                    type=GatewayCommandType.SEND_MESSAGE,
                    call_id=event.call_id,
                    payload={"kind": "local_tool_call", "name": lname},
                ).to_json_bytes())

        await outbox.put(GatewayCommand(
            type=GatewayCommandType.TOOL_CALL_OUTPUT,
            call_id=event.call_id,
            payload={
                "tool_call_id": tool_call_id,
                "output_json": json.dumps(result),
                "is_error": is_error,
            },
        ).to_json_bytes())

    async def start(self, bind: str = "127.0.0.1:8002",
                    server_credentials: grpc.ServerCredentials | None = None) -> None:
        self._server = grpc.aio.server()
        add_bridge_handler(self._server, self._handler)
        if server_credentials is not None:
            self._server.add_secure_port(bind, server_credentials)
        else:
            self._server.add_insecure_port(bind)
        await self._server.start()
        logger.info("business bridge server listening on %s (%s)",
                    bind, "secure" if server_credentials else "insecure")

    async def wait(self) -> None:
        if self._server is not None:
            await self._server.wait_for_termination()

    async def stop(self) -> None:
        if self._server is not None:
            await self._server.stop(grace=1.0)


__all__ = ["BusinessBridgeServer", "BusinessSession"]
