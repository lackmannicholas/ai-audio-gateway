"""The hand-rolled agent abstraction. No SDK.

A ``VoiceAgent`` is the business-plane object that knows three things:

  * its instructions (the system prompt the realtime model runs under),
  * its greeting,
  * the toolset it exposes to the realtime model.

That's it. The agent does not own the audio, the socket, or the realtime model
— those live in the gateway. The agent owns *meaning*: what the assistant is
for, and what it can do.

Both concrete agents (single, responder/thinker) subclass this. The only thing
that differs between them is which tools they expose:

  * SingleVoiceAgent exposes the four café tools directly. The gateway builds
    four proxies; every tool call crosses the wire.
  * ResponderThinkerAgent exposes ONE tool — the thinker — and keeps the four
    café tools behind it, inside the business plane. The gateway builds one
    proxy; only the thinker call crosses the wire, and the thinker's own tool
    calls stay local.

The gateway is identical in both cases. It just gets a different list of
``ToolSpec`` at configure time.
"""

from __future__ import annotations

import abc

from business.tools.base import Toolset


class VoiceAgent(abc.ABC):
    """Base class for a business-plane voice agent."""

    #: Stable identifier reported to the gateway and shown in the UI.
    name: str

    @property
    @abc.abstractmethod
    def instructions(self) -> str:
        """The system prompt the realtime model runs under."""

    @property
    def greeting_instructions(self) -> str:
        return "Greet the caller warmly, say you're the café assistant, then wait."

    @abc.abstractmethod
    def exposed_toolset(self) -> Toolset:
        """The tools the gateway should build proxies for.

        This is the method that distinguishes the two agents. Whatever this
        returns becomes proxy tools on the gateway; everything else the agent
        can do stays inside the business plane.
        """


__all__ = ["VoiceAgent"]
