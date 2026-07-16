"""ToolContext: every tool invocation receives per-call runtime context.

Replaces the old mechanism where the business session poked attributes onto a
specifically-named tool. Now the context is a parameter threaded to every
Tool.invoke — so any tool, not just consult_thinker, can see the call id, the
turn, and a live staleness check.
"""

from __future__ import annotations

from typing import Any

import pytest

from business.grpc_server import BusinessSession
from business.tools.base import Tool, ToolContext, Toolset


class ContextProbe(Tool):
    name = "probe"
    description = "records the context it was invoked with"
    params_json_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def __init__(self) -> None:
        self.seen: ToolContext | None = None

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> Any:
        self.seen = ctx
        return {"stale": ctx.is_stale(), "turn_id": ctx.turn_id, "call_id": ctx.call_id}


@pytest.mark.asyncio
async def test_toolset_threads_context_to_invoke():
    probe = ContextProbe()
    toolset = Toolset([probe])
    ctx = ToolContext(call_id="call_x", turn_id=3, is_stale=lambda: False)

    result = await toolset.invoke("probe", {}, ctx)

    assert probe.seen is ctx
    assert result == {"stale": False, "turn_id": 3, "call_id": "call_x"}


def test_is_stale_tracks_live_turn():
    live = {"turn": 0}
    ctx = ToolContext(call_id="c", turn_id=0, is_stale=lambda: ctx.turn_id != live["turn"])
    assert ctx.is_stale() is False
    live["turn"] = 1  # caller barged in
    assert ctx.is_stale() is True


def test_session_tool_context_goes_stale_on_barge_in():
    # The context a BusinessSession builds must reflect a later barge-in.
    from business.agents.single_agent import SingleVoiceAgent

    session = BusinessSession("call_1", SingleVoiceAgent())
    ctx = session.tool_context(snapshot_turn=0)
    assert ctx.call_id == "call_1"
    assert ctx.is_stale() is False
    session.current_turn_id = 1  # barge-in advanced the live turn
    assert ctx.is_stale() is True
