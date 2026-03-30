"""
Axiom Engine v2.3 — Relevance Ranker (Module 5)

Responsibilities:
  - Ranks scored_chunks by relevance to the user query using
    keyword overlap scoring (deterministic, no LLM calls).
  - Combines relevance score with the upstream quality_score
    for a final ranking_score.
  - Trims to top-N chunks (max_ranked_chunks from pipeline config).
  - Updates GraphState keys: ranked_chunks, audit_trail.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from state import GraphState


# ---------------------------------------------------------------------------
# Text tokenization for keyword matching
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common English stopwords to exclude from relevance scoring.
_STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "between",
    "through", "after", "before", "during", "without", "and", "or", "but",
    "not", "no", "if", "then", "than", "that", "this", "it", "its",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "all", "each", "every", "both", "few", "more", "most", "some",
    "such", "only", "very", "just", "so", "also",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenization with stopword removal."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

def compute_relevance_score(query: str, chunk_text: str) -> float:
    """
    Compute keyword-overlap relevance between a query and a chunk.
    Returns a score in [0.0, 1.0].

    Uses term frequency overlap: fraction of query terms found in chunk.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    chunk_tokens_set = set(_tokenize(chunk_text))
    if not chunk_tokens_set:
        return 0.0

    # Count how many query terms appear in the chunk.
    query_counter = Counter(query_tokens)
    matches = sum(
        count for term, count in query_counter.items()
        if term in chunk_tokens_set
    )

    return round(min(1.0, matches / len(query_tokens)), 4)


# ---------------------------------------------------------------------------
# Combined ranking score
# ---------------------------------------------------------------------------

_RELEVANCE_WEIGHT = 0.6
_QUALITY_WEIGHT = 0.4


def compute_ranking_score(relevance: float, quality: float) -> float:
    """Weighted combination of relevance and quality for final ranking."""
    return round(
        _RELEVANCE_WEIGHT * relevance + _QUALITY_WEIGHT * quality, 4
    )


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------

def _make_audit_event(
    event_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "event_id": str(uuid4()),
        "node": "ranker",
        "event_type": event_type,
        "payload": payload,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RANKED = 10


def ranker_node(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node — Relevance Ranking.

    Reads scored_chunks and user_query, computes relevance scores,
    combines with quality_score, ranks, and trims to top-N.

    Returns keys: ranked_chunks, audit_trail
    """
    audit: list[dict[str, Any]] = []
    scored_chunks: list[dict] = state.get("scored_chunks") or []
    user_query: str = state.get("user_query", "")

    pipeline_cfg: dict = state.get("pipeline_config") or {}
    stages_cfg: dict = pipeline_cfg.get("stages") or {}
    max_ranked: int = stages_cfg.get("max_ranked_chunks", _DEFAULT_MAX_RANKED)

    audit.append(
        _make_audit_event(
            "ranker_start",
            {
                "input_chunk_count": len(scored_chunks),
                "max_ranked_chunks": max_ranked,
            },
        )
    )

    ranked: list[dict[str, Any]] = []
    for chunk in scored_chunks:
        text: str = chunk.get("text", "")
        quality: float = chunk.get("quality_score", 0.5)

        relevance = compute_relevance_score(user_query, text)
        ranking_score = compute_ranking_score(relevance, quality)

        ranked_chunk = {
            **chunk,
            "relevance_score": relevance,
            "ranking_score": ranking_score,
        }
        ranked.append(ranked_chunk)

    # Sort by ranking_score descending, then trim to top-N.
    ranked.sort(key=lambda c: c["ranking_score"], reverse=True)
    trimmed = ranked[:max_ranked]

    audit.append(
        _make_audit_event(
            "ranker_complete",
            {
                "total_scored": len(ranked),
                "returned_top_n": len(trimmed),
                "max_ranked_chunks": max_ranked,
            },
        )
    )

    return {
        "ranked_chunks": trimmed,
        "audit_trail": audit,
    }
