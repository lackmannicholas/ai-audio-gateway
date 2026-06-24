"""Proves the timeline captures the plane boundary and nested fan-out, so the
rendered view actually shows the architecture."""

from __future__ import annotations

from proto_contract.timeline import Timeline


def test_timeline_renders_nested_fanout():
    tl = Timeline()
    tl.record("call", 0, "gateway")
    tl.record("caller_turn", 1, "gateway")
    tl.record("model_response", 2, "gateway")
    tl.record("tool_call:consult_thinker", 3, "gateway")           # crosses wire
    tl.record("business.tool:consult_thinker", 4, "business")      # executes
    tl.record("business.local:get_menu", 5, "business")            # nested, local
    tl.record("business.local:place_order", 5, "business")         # nested, local

    rendered = tl.render()
    lines = rendered.splitlines()
    assert len(lines) == 7

    # The single wire-crossing tool call is at depth 3 (gateway side).
    assert "tool_call:consult_thinker ·gw" in lines[3]
    # Its execution and the nested local calls are on the business side, deeper.
    assert "business.tool:consult_thinker ·bz" in lines[4]
    assert "business.local:get_menu ·bz" in lines[5]
    assert "business.local:place_order ·bz" in lines[6]

    # The nested local calls are indented deeper than the wire call — the tree
    # the gateway is blind to is visibly contained in the business plane.
    assert lines[5].index("business.local") > lines[3].index("tool_call")
