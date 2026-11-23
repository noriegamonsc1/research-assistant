"""LangGraph wiring: a deterministic pipeline with two bounded loops.

  intake -> decompose -> detect_field -> plan_queries -> retrieve --(thin, <=2 rounds)--> plan_queries
                                                            |ok
  landscape -> gaps -> directions -> novelty_check -> critique --(revise, <=1)--> directions
                                                                   |done
                                                   rank -> write -> verify -> render -> END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from . import nodes
from .llm import reset_usage
from .render import render_node
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

    g.set_entry_point("intake")
    for a, b in zip(LINEAR, LINEAR[1:]):
        g.add_edge(a, b)
    g.add_conditional_edges("retrieve", nodes.coverage_ok, {"ok": "landscape", "thin": "plan_queries"})
    for a, b in zip(MIDDLE, MIDDLE[1:]):
        g.add_edge(a, b)
    g.add_conditional_edges("critique", nodes.needs_revision, {"revise": "directions", "done": "rank"})
    for a, b in zip(TAIL, TAIL[1:]):
        g.add_edge(a, b)
    g.add_edge("render", END)
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
