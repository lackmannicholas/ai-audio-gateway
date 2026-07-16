"""Agent B — the responder/thinker agent.

The responder (the realtime model, fast, conversational) is given exactly ONE
tool: ``consult_thinker``. When the conversation needs real reasoning — validate
an order against the menu, price it, place it — the responder calls the thinker
and keeps the caller company while it works.

The thinker is itself a ``Tool``, but a special one: its ``invoke`` runs a
hand-rolled inner agent loop over the *full* café toolset. So when the gateway
relays a single ``consult_thinker`` call across the wire, what happens behind it
in the business plane is a whole tree of tool calls (get_menu, place_order, ...)
that the gateway never sees.

That asymmetry is the entire point of the architecture:

    gateway sees:     one tool call out, one result back
    business plane:   responder -> thinker -> [get_menu, place_order, ...]

The gateway is blind to the tree. The proxy boundary is drawn exactly where the
media plane stops caring.
"""

from __future__ import annotations

from typing import Any

from business.agents.base import VoiceAgent
from business.thinker import Thinker, ThinkerModel
from business.tools.base import Tool, Toolset, ToolContext
from business.tools.cafe_tools import build_cafe_toolset

_RESPONDER_INSTRUCTIONS = """\
You are the voice of a neighborhood café. You are fast and conversational, but
you do NOT decide menu details, prices, or place orders yourself. For anything
that needs a real answer — what's available, what something costs, placing or
checking an order — call consult_thinker with a plain-language description of
what the caller wants, then relay its answer naturally.

While the thinker works, keep the caller company: acknowledge, confirm what you
heard, but don't invent facts. Speak out loud, so keep it short.
"""


class ConsultThinkerTool(Tool):
    """The one tool the responder gets. Behind it: the whole café toolset."""

    name = "consult_thinker"
    description = (
        "Hand off to the thinker for anything requiring real reasoning or "
        "action: checking the menu, pricing, placing an order, or checking "
        "order status. Pass a plain-language description of what the caller "
        "wants. Returns the thinker's resolved answer."
    )
    params_json_schema = {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "Plain-language description of what the caller wants.",
            }
        },
        "required": ["request"],
        "additionalProperties": False,
    }

    def __init__(self, thinker: Thinker) -> None:
        self._thinker = thinker

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> Any:
        request = str(arguments.get("request", "")).strip()
        # This call fans out into multiple café-tool calls inside the business
        # plane. None of them cross the gateway wire. The ctx (turn + staleness)
        # rides along so a barge-in can abandon this work instead of speaking a
        # stale answer over a conversation that has moved on.
        return await self._thinker.run(request, ctx)


class ResponderThinkerAgent(VoiceAgent):
    name = "cafe_responder_thinker"

    def __init__(self, thinker_model: ThinkerModel | None = None) -> None:
        # The thinker reasons over the full café toolset.
        self._cafe_tools = build_cafe_toolset()
        self._thinker = Thinker(toolset=self._cafe_tools, model=thinker_model)
        self._exposed = Toolset([ConsultThinkerTool(self._thinker)])

    @property
    def instructions(self) -> str:
        return _RESPONDER_INSTRUCTIONS

    @property
    def greeting_instructions(self) -> str:
        return "Greet the caller warmly as the café assistant, then wait."

    def exposed_toolset(self) -> Toolset:
        # Just one proxy on the gateway: the thinker. The four café tools live
        # behind it, here in the business plane.
        return self._exposed


__all__ = ["ResponderThinkerAgent", "ConsultThinkerTool"]
