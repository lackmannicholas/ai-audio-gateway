"""The four café tools, written explicitly. Each is a ``Tool`` subclass with a
hand-written JSON schema and an async ``invoke``. No decorators, no signature
introspection — the schema you see is the schema that crosses the wire.
"""

from __future__ import annotations

from typing import Any

from business.domain.cafe import MENU, STORE, STORE_HOURS
from business.tools.base import Tool, Toolset, ToolContext


class GetMenuTool(Tool):
    name = "get_menu"
    description = (
        "List the café menu: every drink, its base price, and the available "
        "sizes and milk options. Call this before placing an order or whenever "
        "the caller asks what's available."
    )
    params_json_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> Any:
        return [
            {
                "name": item.name,
                "price": item.price,
                "sizes": list(item.sizes),
                "milks": list(item.milks),
            }
            for item in MENU.values()
        ]


class GetStoreHoursTool(Tool):
    name = "get_store_hours"
    description = (
        "Get the café's opening hours. Optionally pass a single weekday "
        "(e.g. 'friday') to get just that day; otherwise returns the whole week."
    )
    params_json_schema = {
        "type": "object",
        "properties": {
            "day": {
                "type": "string",
                "description": "Lowercase weekday name, e.g. 'monday'. Optional.",
            }
        },
        "additionalProperties": False,
    }

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> Any:
        day = (arguments.get("day") or "").strip().lower()
        if day:
            return {day: STORE_HOURS.get(day, "closed")}
        return dict(STORE_HOURS)


class PlaceOrderTool(Tool):
    name = "place_order"
    description = (
        "Place a café order. Each line item needs a drink name, size, and milk. "
        "Validate against the menu first with get_menu. Returns an order id and "
        "the total price."
    )
    params_json_schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "drink": {"type": "string"},
                        "size": {"type": "string"},
                        "milk": {"type": "string"},
                    },
                    "required": ["drink", "size"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> Any:
        raw_items = arguments.get("items") or []
        priced: list[dict] = []
        for it in raw_items:
            drink = str(it.get("drink", "")).strip().lower()
            size = str(it.get("size", "medium")).strip().lower()
            milk = str(it.get("milk", "none")).strip().lower()
            if drink not in MENU:
                return {"error": f"'{drink}' is not on the menu", "ok": False}
            price = STORE.price_item(drink, size)
            priced.append({"drink": drink, "size": size, "milk": milk, "price": price})
        order = STORE.place_order(priced)
        return {"ok": True, "order_id": order.order_id, "total": order.total,
                "items": order.items, "status": order.status}


class CheckOrderStatusTool(Tool):
    name = "check_order_status"
    description = "Look up the status and contents of an existing order by its order id."
    params_json_schema = {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
        "additionalProperties": False,
    }

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> Any:
        order = STORE.get_order(str(arguments.get("order_id", "")))
        if order is None:
            return {"found": False}
        return {"found": True, "order_id": order.order_id, "status": order.status,
                "total": order.total, "items": order.items}


def build_cafe_toolset() -> Toolset:
    """The café toolset. Used directly by the single agent, and behind the
    thinker in the responder/thinker agent."""
    return Toolset([
        GetMenuTool(),
        GetStoreHoursTool(),
        PlaceOrderTool(),
        CheckOrderStatusTool(),
    ])


__all__ = ["build_cafe_toolset", "GetMenuTool", "GetStoreHoursTool",
           "PlaceOrderTool", "CheckOrderStatusTool"]
