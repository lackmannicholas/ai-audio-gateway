"""Agent A — the single voice agent.

One brain. Exposes the four café tools directly to the realtime model. Every
tool call the model makes crosses the gRPC wire to the business plane, executes,
and the result crosses back. Flat. Simple. Perfectly adequate for
straightforward ordering.

This is the baseline that makes the responder/thinker contrast legible.
"""

from __future__ import annotations

from business.agents.base import VoiceAgent
from business.tools.base import Toolset
from business.tools.cafe_tools import build_cafe_toolset

_INSTRUCTIONS = """\
You are the voice assistant for a neighborhood café. Be warm, brief, and natural
— you are speaking out loud, so keep replies short.

You can look up the menu, check store hours, place orders, and check order
status using your tools. Always confirm the drink, size, and milk before placing
an order. Read prices back naturally (e.g. "five twenty-five", not "$5.25").
"""


class SingleVoiceAgent(VoiceAgent):
    name = "cafe_single"

    def __init__(self) -> None:
        self._tools = build_cafe_toolset()

    @property
    def instructions(self) -> str:
        return _INSTRUCTIONS

    def exposed_toolset(self) -> Toolset:
        # The gateway builds a proxy for each of these four tools.
        return self._tools


__all__ = ["SingleVoiceAgent"]
