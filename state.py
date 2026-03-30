"""
Axiom Engine v2.3 — LangGraph GraphState
Uses typing.Annotated + operator.add for append-only list fields so that
graph re-entry (override flow) incrementally appends new data rather than
overwriting previously approved state.
"""

from __future__ import annotations

import operator
from typing import Annotated, Sequence, TypedDict


class GraphState(TypedDict):
    """
    Shared mutable state threaded through all LangGraph nodes.

    Fields annotated with `Annotated[Sequence[...], operator.add]` are
    reducers: LangGraph merges node return values by *appending* rather than
    replacing, which is critical for the incremental override logic (§5 of the
    architecture document).
    """

    # ------------------------------------------------------------------
    # INPUT — populated once at graph entry; never mutated by nodes
    # ------------------------------------------------------------------
    request_id: str
    user_query: str
    app_config: dict          # Serialised AppConfig
    models_config: dict       # Serialised ModelConfig
    pipeline_config: dict     # Serialised PipelineConfig

    # ------------------------------------------------------------------
    # RETRIEVAL STATE
    # ------------------------------------------------------------------
    search_queries: list[str]
    # operator.add — source override re-entry appends new chunks without
    # invalidating chunks that already passed quality scoring.
    indexed_chunks: Annotated[Sequence[dict], operator.add]

    # ------------------------------------------------------------------
    # COGNITIVE STATE
    # ------------------------------------------------------------------
    is_answerable: bool
    # Plain list — Synthesizer replaces its output on each rewrite pass.
    draft_sentences: list[dict]

    # ------------------------------------------------------------------
    # VERIFICATION LOOP STATE
    # ------------------------------------------------------------------
    # operator.add — each verification pass appends new rewrite requests
    # so the Synthesizer can see the full correction history.
    rewrite_requests: Annotated[Sequence[str], operator.add]
    # Overwritten each pass — number of NEW rewrite requests from the
    # most recent verification pass. Used by route_post_verification to
    # decide whether to loop (accumulated list is for correction context).
    pending_rewrite_count: int
    # Incremented by the verification node on every loop iteration.
    loop_count: int

    # ------------------------------------------------------------------
    # OUTPUT STATE
    # ------------------------------------------------------------------
    # Plain list — replaced wholesale once verification fully passes.
    final_sentences: list[dict]
    # operator.add — every node appends its own audit events; the audit
    # trail is never overwritten, preserving causality across re-entries.
    audit_trail: Annotated[Sequence[dict], operator.add]


def make_initial_state(
    request_id: str,
    user_query: str,
    app_config: dict,
    models_config: dict,
    pipeline_config: dict,
) -> GraphState:
    """
    Construct a zero-valued GraphState for a fresh pipeline invocation.
    Explicit initialisation of every key prevents KeyError inside nodes.
    """
    return GraphState(
        request_id=request_id,
        user_query=user_query,
        app_config=app_config,
        models_config=models_config,
        pipeline_config=pipeline_config,
        search_queries=[],
        indexed_chunks=[],
        is_answerable=True,
        draft_sentences=[],
        rewrite_requests=[],
        pending_rewrite_count=0,
        loop_count=0,
        final_sentences=[],
        audit_trail=[],
    )
