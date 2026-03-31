"""
Axiom Engine v2.3 — Relevance Ranker (Module 5)

Responsibilities:
  - Ranks scored_chunks by relevance to the user query using BM25-inspired
    scoring (term frequency × inverse document frequency).
  - Combines relevance score with the upstream quality_score
    for a final ranking_score.
  - Trims to top-N chunks (max_ranked_chunks from pipeline config).
  - Updates GraphState keys: ranked_chunks, audit_trail.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from axiom_engine.state import GraphState

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
# BM25 scoring
# ---------------------------------------------------------------------------

# BM25 tuning parameters.
_BM25_K1 = 1.2   # Term frequency saturation
_BM25_B = 0.75   # Length normalization strength


def _compute_idf(term: str, doc_token_sets: list[set[str]], n_docs: int) -> float:
    """Compute inverse document frequency for a term across the corpus."""
    doc_freq = sum(1 for doc_tokens in doc_token_sets if term in doc_tokens)
    if doc_freq == 0:
        return 0.0
    return math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)


def compute_relevance_score(
    query: str,
    chunk_text: str,
    idf_map: dict[str, float] | None = None,
    avg_doc_len: float = 1.0,
) -> float:
    """
    Compute BM25-based relevance between a query and a chunk.
    Returns a score in [0.0, 1.0] (normalized).

    When idf_map is provided, uses BM25 with IDF weighting.
    Otherwise falls back to simple term-frequency overlap.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    chunk_tokens = _tokenize(chunk_text)
    if not chunk_tokens:
        return 0.0

    chunk_tf = Counter(chunk_tokens)
    doc_len = len(chunk_tokens)

    score = 0.0
    query_term_set = set(query_tokens)

    for term in query_term_set:
        tf = chunk_tf.get(term, 0)
        if tf == 0:
            continue

        idf = (idf_map or {}).get(term, 1.0)

        # BM25 term frequency component with length normalization.
        numerator = tf * (_BM25_K1 + 1)
        denominator = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * doc_len / max(avg_doc_len, 1.0))
        score += idf * (numerator / denominator)

    # Normalize to [0, 1] range — divide by theoretical max (all query terms
    # present with high TF and IDF=max_idf).
    max_idf = max((idf_map or {}).values(), default=1.0)
    max_possible = len(query_term_set) * max_idf * (_BM25_K1 + 1) / (1 + _BM25_K1)
    if max_possible > 0:
        score = min(1.0, score / max_possible)

    return round(score, 4)


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

    Reads scored_chunks and user_query, computes BM25 relevance scores
    with IDF weighting across the chunk corpus, combines with quality_score,
    ranks, and trims to top-N.

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

    # Pre-compute IDF across the chunk corpus for BM25 scoring.
    query_tokens = set(_tokenize(user_query))
    doc_token_sets: list[set[str]] = []
    doc_lengths: list[int] = []

    for chunk in scored_chunks:
        tokens = _tokenize(chunk.get("text", ""))
        doc_token_sets.append(set(tokens))
        doc_lengths.append(len(tokens))

    n_docs = len(scored_chunks)
    avg_doc_len = sum(doc_lengths) / n_docs if n_docs > 0 else 1.0

    idf_map: dict[str, float] = {}
    for term in query_tokens:
        idf_map[term] = _compute_idf(term, doc_token_sets, n_docs)

    ranked: list[dict[str, Any]] = []
    for chunk in scored_chunks:
        text: str = chunk.get("text", "")
        quality: float = chunk.get("quality_score", 0.5)

        relevance = compute_relevance_score(user_query, text, idf_map, avg_doc_len)
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
