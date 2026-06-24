"""The explicit ``Tool`` contract — no SDK, no decorator magic.

A tool is three pieces of metadata the realtime model needs in order to call it,
plus one async method that does the work:

    name              what the model calls
    description       the natural-language "when/why" the model reasons about
    params_json_schema  the argument shape the model fills in
    invoke(args)      the actual execution

We write the JSON schema by hand rather than introspecting a function
signature. It is more verbose, but for a reference implementation that verbosity
is the point: there is no hidden layer turning your function into a schema. What
you see is what crosses the wire.

Crucially, ``Tool`` is the *implementation* side. It lives in the business
plane. The gateway never holds a real ``Tool`` — it holds a proxy built from the
tool's ``spec()`` (name + description + schema, no ``invoke``). That separation
is the whole architecture: the schema travels to the media plane, the execution
stays in the meaning plane.
"""

from __future__ import annotations

import abc
from typing import Any

from proto_contract.envelopes import ToolSpec


class Tool(abc.ABC):
    """A callable capability the agent can expose to the realtime model."""

    #: What the realtime model calls.
    name: str
    #: The natural-language description the model reasons over to decide *when*
    #: to call this tool. This is the semantic layer a bare RPC schema lacks.
    description: str
    #: JSON Schema describing the arguments. Written by hand on purpose.
    params_json_schema: dict[str, Any] = {}
    #: Whether the model must conform strictly to the schema.
    strict_json_schema: bool = True

    @abc.abstractmethod
    async def invoke(self, arguments: dict[str, Any]) -> Any:
        """Execute the tool. Runs in the business plane, never on the hot path.

        The return value must be JSON-serializable; it is what travels back
        across the wire to the gateway proxy and then to the realtime model.
        """

    def spec(self) -> ToolSpec:
        """The wire-safe description handed to the gateway at configure time.

        Note what is absent: ``invoke``. The gateway gets the shape, not the
        behavior.
        """
        return ToolSpec(
            name=self.name,
            description=self.description,
            params_json_schema=self.params_json_schema,
            strict_json_schema=self.strict_json_schema,
        )


class Toolset:
    """A named collection of tools, looked up by name at execution time.

    This is also the sanitization / swap seam. In the reference repo it is
    backed by in-memory café tools. In a real system the same interface could
    be backed by MCP servers — the gateway would not know the difference,
    because all it ever sees is the list of ``ToolSpec`` this produces.
    """

    def __init__(self, tools: list[Tool]) -> None:
        self._by_name: dict[str, Tool] = {t.name: t for t in tools}

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._by_name.values()]

    def names(self) -> list[str]:
        return list(self._by_name)

    def get(self, name: str) -> Tool:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"unknown tool: {name!r}") from None

    async def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self.get(name).invoke(arguments)


__all__ = ["Tool", "Toolset"]
