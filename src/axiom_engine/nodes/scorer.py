"""
Axiom Engine v2.3 — Source & Chunk Quality Scorer (Modules 3–4)

Responsibilities:
  - Assigns a source_quality_score (0.0–1.0) to each chunk based on domain
    authority heuristics. Domain lists are configurable via app_config and
    fall back to built-in defaults.
  - Assigns a chunk_quality_score (0.0–1.0) based on content quality signals
    (length, information density, presence of data markers).
  - Computes a combined quality_score as a weighted blend of both signals.
  - Filters out chunks below a minimum quality threshold.
  - Updates GraphState keys: scored_chunks, audit_trail.

Both scorers are deterministic — no LLM calls required.
"""

from __future__ import annotations

import logging
import re
from functools import partial
from typing import Any

from axiom_engine.state import GraphState
from axiom_engine.utils.audit import make_audit_event

_audit = partial(make_audit_event, "scorer")
logger = logging.getLogger("axiom_engine.scorer")

# ---------------------------------------------------------------------------
# Source quality — built-in defaults (overridable via app_config)
# ---------------------------------------------------------------------------

_DEFAULT_AUTHORITATIVE_DOMAINS: set[str] = {
    # Academic / government
    "arxiv.org",
    "scholar.google.com",
    "pubmed.ncbi.nlm.nih.gov",
    "nature.com",
    "science.org",
    "ieee.org",
    "acm.org",
    "nih.gov",
    "cdc.gov",
    "who.int",
    "europa.eu",
    # Reference
    "en.wikipedia.org",
    "britannica.com",
    # Major tech docs
    "docs.python.org",
    "developer.mozilla.org",
}

_DEFAULT_LOW_QUALITY_DOMAINS: set[str] = {
    "reddit.com",
    "quora.com",
    "answers.yahoo.com",
    "medium.com",
    "blogspot.com",
}

_DEFAULT_SOURCE_SCORE = 0.5


def build_domain_sets(
    app_config: dict[str, Any],
) -> tuple[set[str], set[str]]:
    """
    Build authoritative and low-quality domain sets.
    Uses app_config overrides if present, otherwise falls back to defaults.
    """
    authoritative = set(app_config.get("authoritative_domains", []))
    low_quality = set(app_config.get("low_quality_domains", []))

    # Merge with defaults (config additions are additive).
    authoritative = authoritative | _DEFAULT_AUTHORITATIVE_DOMAINS
    low_quality = low_quality | _DEFAULT_LOW_QUALITY_DOMAINS

    return authoritative, low_quality


def is_authoritative_domain(domain: str, authoritative: set[str]) -> bool:
    """Return True when the domain is authoritative or a subdomain of one."""
    domain_lower = domain.lower().strip()
    if domain_lower in authoritative:
        return True
    return any(domain_lower.endswith("." + auth) for auth in authoritative)


def score_source_quality(
    domain: str,
    authoritative: set[str] | None = None,
    low_quality: set[str] | None = None,
) -> float:
    """
    Score a domain's authority on a 0.0–1.0 scale.
      - Authoritative → 0.9
      - Low quality    → 0.3
      - Unknown        → 0.5
    """
    if authoritative is None:
        authoritative = _DEFAULT_AUTHORITATIVE_DOMAINS
    if low_quality is None:
        low_quality = _DEFAULT_LOW_QUALITY_DOMAINS

    domain_lower = domain.lower().strip()
    if domain_lower in authoritative:
        return 0.9
    if domain_lower in low_quality:
        return 0.3
    # Check if domain is a subdomain of an authoritative domain.
    for auth in authoritative:
        if domain_lower.endswith("." + auth):
            return 0.85
    for low in low_quality:
        if domain_lower.endswith("." + low):
            return 0.3
    return _DEFAULT_SOURCE_SCORE


# ---------------------------------------------------------------------------
# Chunk quality heuristics
# ---------------------------------------------------------------------------

# Regex patterns that signal information-rich content.
_DATA_MARKERS = re.compile(
    r"\d+\.?\d*\s*%"  # Percentages
    r"|\d{4}"  # Years / large numbers
    r"|(?:fig(?:ure)?|table)\s*\d"  # Figure/table references
    r"|https?://",  # Embedded URLs (citations within text)
    re.IGNORECASE,
)

_MIN_QUALITY_THRESHOLD = 0.2


def score_chunk_quality(text: str) -> float:
    """
    Score a chunk's content quality on 0.0–1.0 based on:
      - Length (longer paragraphs tend to carry more information)
      - Information density (presence of numbers, data markers)
    """
    if not text or not text.strip():
        return 0.0

    length = len(text.strip())

    # Length score: ramps from 0.2 at 40 chars to 1.0 at 500+ chars.
    length_score = min(1.0, 0.2 + (length - 40) * 0.8 / 460) if length >= 40 else 0.1

    # Data marker density bonus.
    markers = _DATA_MARKERS.findall(text)
    density_bonus = min(0.3, len(markers) * 0.1)

    return round(min(1.0, length_score + density_bonus), 4)


# ---------------------------------------------------------------------------
# Combined scoring
# ---------------------------------------------------------------------------

_SOURCE_WEIGHT = 0.4
_CHUNK_WEIGHT = 0.6


def compute_combined_score(source_score: float, chunk_score: float) -> float:
    """Weighted combination of source and chunk quality scores."""
    return round(_SOURCE_WEIGHT * source_score + _CHUNK_WEIGHT * chunk_score, 4)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def scorer_node(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node — Source & Chunk Quality Scoring.

    Reads indexed_chunks, assigns source and chunk quality scores,
    computes a combined quality_score, and filters below threshold.
    Domain lists are configurable via app_config keys:
      - authoritative_domains: list[str]
      - low_quality_domains: list[str]

    Returns keys: scored_chunks, audit_trail
    """
    audit: list[dict[str, Any]] = []
    indexed_chunks: list[dict] = list(state.get("indexed_chunks") or [])
    app_cfg: dict = state.get("app_config") or {}

    authoritative, low_quality = build_domain_sets(app_cfg)

    audit.append(
        _audit(
            "scorer_start",
            {"input_chunk_count": len(indexed_chunks)},
        )
    )

    scored: list[dict[str, Any]] = []
    filtered_count = 0

    for chunk in indexed_chunks:
        domain: str = chunk.get("domain", "")
        text: str = chunk.get("text", "")

        source_score = score_source_quality(domain, authoritative, low_quality)
        chunk_score = score_chunk_quality(text)
        combined = compute_combined_score(source_score, chunk_score)

        if combined < _MIN_QUALITY_THRESHOLD:
            filtered_count += 1
            continue

        scored_chunk = {
            **chunk,
            "source_quality_score": source_score,
            "chunk_quality_score": chunk_score,
            "quality_score": combined,
        }
        scored.append(scored_chunk)

    # Sort by quality_score descending for downstream consumption.
    scored.sort(key=lambda c: c["quality_score"], reverse=True)

    audit.append(
        _audit(
            "scorer_complete",
            {
                "input_chunks": len(indexed_chunks),
                "scored_chunks": len(scored),
                "filtered_below_threshold": filtered_count,
            },
        )
    )

    return {
        "scored_chunks": scored,
        "audit_trail": audit,
    }
