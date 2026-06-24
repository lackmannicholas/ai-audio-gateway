"""The thinker: a hand-rolled, SDK-free inner agent loop.

When the responder calls ``consult_thinker``, this runs: a classic tool-calling
loop. Ask the model what to do; if it wants a tool, run the tool (in the
business plane), feed the result back; repeat until the model produces a final
answer. This is the loop the big SDKs hide — here it is in ~40 lines so you can
see exactly what "an agent" is.

The model is pluggable via ``ThinkerModel``:

  * ``MockThinkerModel`` (default) — deterministic, zero-dependency. It pattern-
    matches the request and emits a scripted sequence of tool calls so the demo
    runs with no API key and no cost. It is NOT an LLM; it is a stand-in that
    exercises the same loop a real model would drive.
  * ``OpenAIThinkerModel`` — used when THINKER_BACKEND=openai. Real chat-
    completions tool calling. (Implemented but optional; the import is lazy.)

The point of this module for the article: the *loop* is the agent. The model is
swappable. And every tool call here stays inside the business plane — the
gateway never sees them.
"""

from __future__ import annotations

import abc
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from business.tools.base import Toolset


# --------------------------------------------------------------------------- #
# Model abstraction.
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass
class ModelStep:
    """One model turn: either it wants tools, or it's done with a final answer."""

    tool_calls: list[ToolCall]
    final_text: str | None


class ThinkerModel(abc.ABC):
    @abc.abstractmethod
    async def step(self, request: str, history: list[dict]) -> ModelStep:
        """Given the request and the running history, decide the next step."""


class MockThinkerModel(ThinkerModel):
    """Deterministic stand-in. Drives the real loop without an LLM.

    Strategy: on the first step, decide which tools the request implies and
    emit them. On the second step (after tool results are in history), produce
    a final natural-language answer from those results.
    """

    async def step(self, request: str, history: list[dict]) -> ModelStep:
        r = request.lower()
        already_called = {h["name"] for h in history if h.get("role") == "tool"}

        # Second pass: results are in, summarize.
        if already_called:
            return ModelStep(tool_calls=[], final_text=self._summarize(history))

        # First pass: choose tools from the request.
        calls: list[ToolCall] = []
        n = 0
        if any(w in r for w in ("menu", "have", "available", "offer", "what kind")):
            calls.append(ToolCall("get_menu", {}, f"mock_{n}")); n += 1
        if any(w in r for w in ("hour", "open", "close", "when")):
            calls.append(ToolCall("get_store_hours", {}, f"mock_{n}")); n += 1
        if any(w in r for w in ("order", "get", "want", "latte", "coffee",
                                "cappuccino", "cold brew", "espresso", "i'll have")):
            # A realistic order: validate menu then place it.
            if not any(c.name == "get_menu" for c in calls):
                calls.append(ToolCall("get_menu", {}, f"mock_{n}")); n += 1
            calls.append(ToolCall(
                "place_order",
                {"items": [{"drink": "latte", "size": "large", "milk": "oat"}]},
                f"mock_{n}",
            )); n += 1
        if "status" in r or "ready" in r:
            calls.append(ToolCall("check_order_status", {"order_id": "ord_unknown"}, f"mock_{n}")); n += 1

        if not calls:
            return ModelStep(tool_calls=[], final_text=(
                "I can help with the menu, hours, placing an order, or checking "
                "an order's status. What would you like?"))
        return ModelStep(tool_calls=calls, final_text=None)

    def _summarize(self, history: list[dict]) -> str:
        for h in history:
            if h.get("role") == "tool" and h.get("name") == "place_order":
                data = h.get("content_parsed") or {}
                if data.get("ok"):
                    total = data.get("total")
                    return (f"Order in! That's {self._money(total)}. "
                            f"Your order id is {data.get('order_id')}.")
        for h in history:
            if h.get("role") == "tool" and h.get("name") == "get_menu":
                items = h.get("content_parsed") or []
                names = ", ".join(i["name"] for i in items[:4])
                return f"We've got {names}, and more. What sounds good?"
            if h.get("role") == "tool" and h.get("name") == "get_store_hours":
                return "Here are our hours — anything else?"
        return "Done — anything else?"

    @staticmethod
    def _money(v: Any) -> str:
        try:
            return f"${float(v):.2f}"
        except Exception:
            return str(v)


class OpenAIThinkerModel(ThinkerModel):
    """Real OpenAI chat-completions tool calling. Used when THINKER_BACKEND=openai."""

    def __init__(self, toolset: Toolset, model: str = "gpt-4o-mini") -> None:
        from openai import AsyncOpenAI  # lazy: only needed in this mode
        self._client = AsyncOpenAI()
        self._model = model
        self._tool_schemas = [
            {"type": "function",
             "function": {"name": s.name, "description": s.description,
                          "parameters": s.params_json_schema}}
            for s in toolset.specs()
        ]

    async def step(self, request: str, history: list[dict]) -> ModelStep:
        messages = [{"role": "system",
                     "content": "You are the café thinker. Use tools to answer, "
                                "then give a short spoken-style final answer."},
                    {"role": "user", "content": request}]
        for h in history:
            if h.get("role") == "tool":
                messages.append({"role": "assistant", "content": None,
                                 "tool_calls": [{"id": h["call_id"], "type": "function",
                                                 "function": {"name": h["name"],
                                                              "arguments": h.get("arguments_json", "{}")}}]})
                messages.append({"role": "tool", "tool_call_id": h["call_id"],
                                 "content": h["content"]})
        resp = await self._client.chat.completions.create(
            model=self._model, messages=messages, tools=self._tool_schemas)
        msg = resp.choices[0].message
        if msg.tool_calls:
            calls = [ToolCall(tc.function.name,
                              json.loads(tc.function.arguments or "{}"), tc.id)
                     for tc in msg.tool_calls]
            return ModelStep(tool_calls=calls, final_text=None)
        return ModelStep(tool_calls=[], final_text=msg.content or "")


def _default_model(toolset: Toolset) -> ThinkerModel:
    if os.getenv("THINKER_BACKEND", "mock").lower() == "openai":
        return OpenAIThinkerModel(toolset)
    return MockThinkerModel()


# --------------------------------------------------------------------------- #
# The loop.
# --------------------------------------------------------------------------- #
class Thinker:
    """A hand-rolled tool-calling agent loop over a toolset.

    Emits per-tool-call notifications via ``on_tool_call`` so the UI / tracing
    can show the nested fan-out that stays inside the business plane.
    """

    def __init__(self, toolset: Toolset, model: ThinkerModel | None = None,
                 max_steps: int = 4) -> None:
        self._toolset = toolset
        self._model = model or _default_model(toolset)
        self._max_steps = max_steps
        # Optional async callback: (tool_name, arguments) -> None, set by the
        # business session so nested calls can be surfaced to the gateway/UI as
        # "local" events.
        self.on_tool_call = None

    async def run(self, request: str,
                  turn_id: int = 0,
                  is_stale: "Callable[[int], bool] | None" = None) -> dict[str, Any]:
        """Run the tool-calling loop.

        ``turn_id`` is the turn this thinker run belongs to (snapshotted from the
        gateway at call time). ``is_stale(turn_id)`` is checked before and after
        each slow step; if it returns True, the conversation has moved on (the
        caller barged in) and we abandon the work rather than return an answer
        that would be spoken over a stale context.
        """
        history: list[dict] = []

        def _stale() -> bool:
            return is_stale(turn_id) if is_stale is not None else False

        for _ in range(self._max_steps):
            if _stale():
                return {"stale": True, "turn_id": turn_id}
            step = await self._model.step(request, history)
            if step.final_text is not None and not step.tool_calls:
                if _stale():
                    return {"stale": True, "turn_id": turn_id}
                return {"answer": step.final_text}
            for call in step.tool_calls:
                if _stale():
                    return {"stale": True, "turn_id": turn_id}
                if self.on_tool_call is not None:
                    await self.on_tool_call(call.name, call.arguments)
                result = await self._toolset.invoke(call.name, call.arguments)
                history.append({
                    "role": "tool",
                    "name": call.name,
                    "call_id": call.call_id,
                    "arguments_json": json.dumps(call.arguments),
                    "content": json.dumps(result),
                    "content_parsed": result,
                })
        # Ran out of steps; return whatever we have.
        return {"answer": "Sorry, I couldn't complete that — could you rephrase?"}


__all__ = ["Thinker", "ThinkerModel", "MockThinkerModel", "OpenAIThinkerModel",
           "ToolCall", "ModelStep"]
