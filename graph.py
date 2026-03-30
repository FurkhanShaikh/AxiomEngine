"""
Axiom Engine v2.3 — LangGraph DAG Compilation (The Engine Core)

Wires the nodes and conditional edges into an executable StateGraph.

DAG topology:
  retriever → synthesizer → verifier ─┐
                 ▲                     │
                 └── (rewrite loop) ◄──┘  (if Tier 4/5 failures & loop < 3)
                                       │
                                       └──► END  (all passed / unanswerable / loop exhausted)
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from nodes.synthesizer import synthesizer_node
from nodes.verification import verification_node
from state import GraphState


# ---------------------------------------------------------------------------
# Conditional edge — the verification loop (LLD §4)
# ---------------------------------------------------------------------------

def route_post_verification(
    state: GraphState,
) -> Literal["synthesizer", "__end__"]:
    """
    Determine whether the graph terminates or loops back to the Synthesizer
    for a rewrite pass.

    Routing rules (architecture §5, LLD §4):
      1. If is_answerable is False → END (escape hatch or insufficient data).
      2. If loop_count >= max_rewrite_loops (default 3) → END (exhaustion).
      3. If rewrite_requests is non-empty → loop back to "synthesizer".
      4. Otherwise → END (all citations verified successfully).
    """
    # Rule 1: escape hatch
    if not state.get("is_answerable", True):
        return "__end__"

    # Rule 2: loop exhaustion
    pipeline_cfg: dict = state.get("pipeline_config") or {}
    stages_cfg: dict = pipeline_cfg.get("stages") or {}
    max_loops: int = stages_cfg.get("max_rewrite_loops", 3)

    if state.get("loop_count", 0) >= max_loops:
        return "__end__"

    # Rule 3: rewrite needed (check THIS pass's failures, not the accumulated history)
    if state.get("pending_rewrite_count", 0) > 0:
        return "synthesizer"

    # Rule 4: all good
    return "__end__"


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def build_axiom_graph() -> StateGraph:
    """
    Construct and compile the Axiom Engine LangGraph DAG.

    Returns the compiled graph, ready to be invoked with an initial state.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("verifier", verification_node)

    # Linear edges
    workflow.set_entry_point("synthesizer")
    workflow.add_edge("synthesizer", "verifier")

    # Conditional edge — the verification loop
    workflow.add_conditional_edges(
        "verifier",
        route_post_verification,
        {
            "synthesizer": "synthesizer",
            "__end__": END,
        },
    )

    return workflow.compile()


# Compiled engine — importable as `from graph import axiom_engine`
axiom_engine = build_axiom_graph()
