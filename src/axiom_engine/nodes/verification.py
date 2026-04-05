"""
Axiom Engine v2.3 — Unified Verification Node (Module 7)

Orchestrates the two-stage verification pipeline:
  Stage 1: MechanicalVerifier (deterministic, non-negotiable)
  Stage 2: SemanticVerifier (configurable LLM check)

This is the single LangGraph node registered as "verifier" in the DAG.
It runs mechanical verification on every citation first, then passes
mechanically-approved citations through semantic verification.

Tier 5 (Hallucinated) citations generate rewrite_requests for the
Synthesizer loop. Tier 4 (Misrepresented) citations also generate
rewrite requests via the semantic verifier.

Updates GraphState keys: final_sentences, rewrite_requests, loop_count,
mechanical_results, audit_trail
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, cast

from axiom_engine.nodes.semantic import semantic_verifier_node
from axiom_engine.state import GraphState
from axiom_engine.utils.audit import make_audit_event
from axiom_engine.verifiers.mechanical import MechanicalVerifier

# Module-level singleton — stateless, safe to reuse.
_mechanical = MechanicalVerifier()
_audit = partial(make_audit_event, "verifier")
logger = logging.getLogger("axiom_engine.verifier")


def _build_tier5_rewrite_request(
    sentence_id: str,
    citation_id: str,
    chunk_id: str,
    failure_reason: str,
) -> str:
    return (
        f"Sentence {sentence_id}, citation {citation_id} (chunk {chunk_id}): "
        f"Tier 5 (hallucinated) failure — {failure_reason}"
    )


def verification_node(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node — Unified Verification.

    Stage 1: Runs MechanicalVerifier on every citation.
    Stage 2: Delegates to semantic_verifier_node for mechanically-passed citations.

    Returns keys: final_sentences, rewrite_requests, loop_count,
                  mechanical_results, audit_trail
    """
    audit: list[dict[str, Any]] = []
    draft_sentences: list[dict] = list(state.get("draft_sentences") or [])
    indexed_chunks: list[dict] = list(state.get("indexed_chunks") or [])
    chunk_lookup: dict[str, dict] = {c["chunk_id"]: c for c in indexed_chunks}

    audit.append(
        _audit(
            "verification_start",
            {
                "sentence_count": len(draft_sentences),
                "loop_count": state.get("loop_count", 0),
            },
        )
    )

    # ------------------------------------------------------------------
    # Stage 1: Mechanical Verification
    # ------------------------------------------------------------------
    mechanical_results: dict[str, str] = {}
    mechanical_rewrite_requests: list[str] = []

    for sentence_dict in draft_sentences:
        sentence_id: str = sentence_dict["sentence_id"]
        citations: list[dict] = sentence_dict.get("citations") or []

        for citation in citations:
            cit_id: str = citation["citation_id"]
            chunk_id: str = citation["chunk_id"]
            exact_quote: str = citation.get("exact_source_quote", "")

            chunk_data = chunk_lookup.get(chunk_id)
            if chunk_data is None:
                # Chunk not found — treat as Tier 5.
                mechanical_results[cit_id] = "failed"
                mechanical_rewrite_requests.append(
                    _build_tier5_rewrite_request(
                        sentence_id,
                        cit_id,
                        chunk_id,
                        f"Chunk {chunk_id} not found in indexed_chunks.",
                    )
                )
                audit.append(
                    _audit(
                        "mechanical_chunk_not_found",
                        {"citation_id": cit_id, "chunk_id": chunk_id},
                    )
                )
                continue

            chunk_text: str = chunk_data.get("text", "")
            result = _mechanical.verify(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                llm_quote=exact_quote,
            )

            mechanical_results[cit_id] = result.status
            audit.append(
                _audit(
                    "mechanical_result",
                    result.audit_proof,
                )
            )

            if result.status == "failed":
                mechanical_rewrite_requests.append(
                    _build_tier5_rewrite_request(
                        sentence_id,
                        cit_id,
                        chunk_id,
                        result.audit_proof.get(
                            "failure_reason",
                            "Normalized quote not found in chunk.",
                        ),
                    )
                )

    audit.append(
        _audit(
            "mechanical_phase_complete",
            {
                "total_citations": len(mechanical_results),
                "passed": sum(1 for v in mechanical_results.values() if v == "passed"),
                "failed": sum(1 for v in mechanical_results.values() if v == "failed"),
            },
        )
    )

    # ------------------------------------------------------------------
    # Stage 2: Semantic Verification (delegates to semantic_verifier_node)
    # ------------------------------------------------------------------
    # Build an intermediate state with mechanical_results injected so the
    # semantic node knows which citations to skip.
    semantic_input_state = cast(GraphState, {**state, "mechanical_results": mechanical_results})

    semantic_result = semantic_verifier_node(semantic_input_state)

    # ------------------------------------------------------------------
    # Merge results
    # ------------------------------------------------------------------
    # Combine mechanical rewrite requests (Tier 5) with semantic ones (Tier 4).
    all_rewrite_requests: list[str] = mechanical_rewrite_requests + semantic_result.get(
        "rewrite_requests", []
    )
    pending_count = len(all_rewrite_requests)

    # Merge audit trails.
    all_audit: list[dict] = audit + semantic_result.get("audit_trail", [])

    return {
        "final_sentences": semantic_result.get("final_sentences", []),
        "rewrite_requests": all_rewrite_requests,
        "pending_rewrite_count": pending_count,
        "loop_count": semantic_result.get("loop_count", state.get("loop_count", 0) + 1),
        "mechanical_results": mechanical_results,
        "audit_trail": all_audit,
    }
