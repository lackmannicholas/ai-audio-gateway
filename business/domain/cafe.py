"""A tiny in-memory café domain. Deliberately trivial — the domain is not the
point, the architecture is. But it needs to be real enough that tools have
something to read and write, and that the thinker has genuine multi-step work
to do (validate an order against the menu, price it, place it).
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MenuItem:
    name: str
    price: float
    sizes: tuple[str, ...] = ("small", "medium", "large")
    milks: tuple[str, ...] = ("whole", "oat", "almond", "none")


MENU: dict[str, MenuItem] = {
    "latte": MenuItem("latte", 4.25),
    "cappuccino": MenuItem("cappuccino", 4.00),
    "cold brew": MenuItem("cold brew", 3.75, milks=("oat", "almond", "none")),
    "drip coffee": MenuItem("drip coffee", 2.50, milks=("whole", "oat", "none")),
    "espresso": MenuItem("espresso", 2.75, sizes=("single", "double"), milks=("none",)),
}

SIZE_UPCHARGE = {"small": 0.0, "medium": 0.50, "large": 1.00,
                 "single": 0.0, "double": 0.75}

STORE_HOURS = {
    "monday": "6:00 AM – 7:00 PM", "tuesday": "6:00 AM – 7:00 PM",
    "wednesday": "6:00 AM – 7:00 PM", "thursday": "6:00 AM – 7:00 PM",
    "friday": "6:00 AM – 8:00 PM", "saturday": "7:00 AM – 8:00 PM",
    "sunday": "7:00 AM – 5:00 PM",
}


@dataclass
class Order:
    order_id: str
    items: list[dict] = field(default_factory=list)
    total: float = 0.0
    status: str = "received"


class CafeStore:
    """Thread-safe in-memory store. One instance per process is fine for a POC."""

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self._lock = threading.Lock()

    def price_item(self, item: str, size: str) -> float:
        menu_item = MENU[item]
        return round(menu_item.price + SIZE_UPCHARGE.get(size, 0.0), 2)

    def place_order(self, items: list[dict]) -> Order:
        order_id = "ord_" + uuid.uuid4().hex[:8]
        total = round(sum(i.get("price", 0.0) for i in items), 2)
        order = Order(order_id=order_id, items=items, total=total, status="received")
        with self._lock:
            self._orders[order_id] = order
        return order

    def get_order(self, order_id: str) -> Order | None:
        with self._lock:
            return self._orders.get(order_id)


# Process-wide store for the POC.
STORE = CafeStore()

__all__ = ["MENU", "MenuItem", "STORE_HOURS", "Order", "CafeStore", "STORE",
           "SIZE_UPCHARGE"]
