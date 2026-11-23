"""LangGraph wiring: a deterministic pipeline with two bounded loops.

  intake -> decompose -> detect_field -> plan_queries -> retrieve --(thin, <=2 rounds)--> plan_queries
                                                            |ok          \\--(insufficient)--> stop_early -> END
  landscape -> gaps -> directions -> novelty_check -> critique --(revise, <=1)--> directions
                                                                   |done
                                                   rank -> write -> verify -> render -> END

Gates (config/rubric.yaml -> gates):
  - insufficient: fewer than min_ledger_to_continue papers after all search rounds -> short report, stop.
  - revise: only when fewer than 3 directions are usable, at most once, never after the cost cap.
  - critique runs at medium effort when few candidates remain; past the cost cap every call steps down.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from . import nodes
from .llm import reset_usage
from .render import render_node, stop_early_node
from .state import State
from .verify import verify_node

LINEAR = ["intake", "decompose", "detect_field", "plan_queries", "retrieve"]
MIDDLE = ["landscape", "gaps", "directions", "novelty_check", "critique"]
TAIL = ["rank", "write", "verify", "render"]


def build():
    g = StateGraph(State)
    for name in LINEAR + MIDDLE + ["rank", "write"]:
        g.add_node(name, getattr(nodes, name))
    g.add_node("verify", verify_node)
    g.add_node("render", render_node)
    g.add_node("stop_early", stop_early_node)

    g.set_entry_point("intake")
    for a, b in zip(LINEAR, LINEAR[1:]):
        g.add_edge(a, b)
    g.add_conditional_edges("retrieve", nodes.coverage_ok, {"ok": "landscape", "thin": "plan_queries", "insufficient": "stop_early"})
    for a, b in zip(MIDDLE, MIDDLE[1:]):
        g.add_edge(a, b)
    g.add_conditional_edges("critique", nodes.needs_revision, {"revise": "directions", "done": "rank"})
    for a, b in zip(TAIL, TAIL[1:]):
        g.add_edge(a, b)
    g.add_edge("render", END)
    g.add_edge("stop_early", END)
    return g.compile()


def run(raw_input: str, user_field: str | None = None, stream: bool = True) -> State:
    reset_usage()
    app = build()
    init: State = {"raw_input": raw_input, "user_field": user_field, "notes": []}
    if not stream:
        return app.invoke(init, {"recursion_limit": 60})
    state: dict = dict(init)
    for update in app.stream(init, {"recursion_limit": 60}, stream_mode="updates"):
        for node, delta in update.items():
            print(f"  ✓ {node}", flush=True)
            for k, v in (delta or {}).items():
                state[k] = state.get(k, []) + v if k == "notes" else v
    return state  # type: ignore[return-value]
