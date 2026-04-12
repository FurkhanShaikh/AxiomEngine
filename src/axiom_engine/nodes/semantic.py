"""
Axiom Engine v2.3 — Semantic Verifier Node (Module 7, Stage 2)

Responsibilities:
  - Runs only after Mechanical Verification has checked every citation.
  - Uses a lightweight LLM to decide whether each mechanically-valid claim
    faithfully represents its cited source chunk in context.
  - Emits citation-level verification objects and sentence-level rollups.
  - Assigns Tier 1 and Tier 2 only from deterministic source signals:
      * Tier 1: at least one authoritative source and no verification failures.
      * Tier 2: multiple independent domains and no verification failures.
      * Tier 3: mechanically valid but authority/consensus not proven.
      * Tier 4: semantic misrepresentation.
      * Tier 5: mechanical failure or unsupported uncited sentence.
  - Never guesses Tier 6 without explicit contradiction logic.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from functools import partial
from typing import Any

import litellm

from axiom_engine.config.observability import LLM_CALL_DURATION, get_tracer
from axiom_engine.models import (
    Citation,
    FinalSentence,
    VerificationResult,
    VerifiedCitation,
)
from axiom_engine.nodes.scorer import build_domain_sets, is_authoritative_domain
from axiom_engine.state import GraphState
from axiom_engine.utils.audit import make_audit_event
from axiom_engine.utils.llm import build_completion_kwargs

logger = logging.getLogger("axiom_engine.semantic")
_audit = partial(make_audit_event, "semantic_verifier")

_MAX_CONCURRENT = int(os.environ.get("AXIOM_MAX_CONCURRENT_LLM", "5"))
_llm_semaphore = threading.Semaphore(_MAX_CONCURRENT)


_SYSTEM_PROMPT = """\
You are the Semantic Verifier for the Axiom Engine. Your job is to assess \
whether a cited claim faithfully represents its source chunk.

You will be given:
  - CLAIM: one sentence the Synthesizer produced
  - QUOTE: the exact substring the Synthesizer cited
  - CHUNK_TEXT: the full source paragraph the quote was taken from

You must respond with a single valid JSON object (no markdown fences):

{
  "semantic_check": "passed" | "failed",
  "failure_reason": "<string if failed, else null>",
  "reasoning": "<one sentence explaining your decision>"
}

RULES:
  - Return semantic_check="passed" only when the claim faithfully represents
    the quoted text in the context of the full chunk.
  - Return semantic_check="failed" when the claim overstates, cherry-picks,
    strips critical context, or otherwise distorts what the chunk says.
  - failure_reason must be specific when semantic_check="failed".
  - Do not infer source authority, consensus, or contradiction tiers.
  - Do NOT wrap your JSON in markdown code fences.
"""

_USER_PROMPT_TEMPLATE = """\
CLAIM:
{claim}

QUOTE:
{quote}

CHUNK_TEXT (full source paragraph):
{chunk_text}

SOURCE METADATA:
{source_metadata}

Assess the claim and output valid JSON only.
"""


def _parse_semantic_response(raw: str) -> dict[str, Any]:
    """
    Parse and validate the semantic verifier's JSON response.
    Strips accidental markdown fences.
    Raises ValueError on parse or schema errors.
    """
    clean = re.sub(r"<think>.*?</think>", "", raw.strip(), flags=re.DOTALL)
    clean = re.sub(r"^```(?:json)?\s*", "", clean.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean.strip())

    try:
        data: dict[str, Any] = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Semantic verifier response is not valid JSON: {exc}") from exc

    if "tier" in data:
        raise ValueError("Semantic verifier response must not include a tier field")

    if data.get("semantic_check") not in ("passed", "failed"):
        raise ValueError(
            f"semantic_check must be 'passed' or 'failed', got {data.get('semantic_check')!r}"
        )

    failure_reason = data.get("failure_reason")
    if data["semantic_check"] == "failed" and not failure_reason:
        raise ValueError("failure_reason is required when semantic_check='failed'")

    return data


def _build_tier4_rewrite_request(
    sentence_id: str,
    citation_id: str,
    chunk_id: str,
    failure_reason: str,
) -> str:
    return (
        f"Sentence {sentence_id}, citation {citation_id} (chunk {chunk_id}): "
        f"Tier 4 (misrepresented) failure — {failure_reason}"
    )


def _build_uncited_sentence_request(sentence_id: str) -> str:
    return (
        f"Sentence {sentence_id}: unsupported sentence — every answer sentence "
        "must include at least one citation with an exact source quote."
    )


def _degraded_verification(reason: str) -> VerificationResult:
    """Tier 3 fallback used when semantic verification is disabled or errors."""
    return VerificationResult(
        tier=3,
        tier_label="model_assisted",
        mechanical_check="passed",
        semantic_check="skipped",
        failure_reason=reason,
    )


def _passed_verification(domain: str, authoritative: set[str]) -> VerificationResult:
    """Build the citation-level verification for a semantically faithful citation."""
    if is_authoritative_domain(domain, authoritative):
        return VerificationResult(
            tier=1,
            tier_label="authoritative",
            mechanical_check="passed",
            semantic_check="passed",
            failure_reason=None,
        )
    return VerificationResult(
        tier=3,
        tier_label="model_assisted",
        mechanical_check="passed",
        semantic_check="passed",
        failure_reason=None,
    )


def _failed_semantic_verification(failure_reason: str) -> VerificationResult:
    """Build the citation-level verification for a semantic misrepresentation."""
    return VerificationResult(
        tier=4,
        tier_label="misrepresented",
        mechanical_check="passed",
        semantic_check="failed",
        failure_reason=failure_reason,
    )


def _aggregate_sentence_verification(
    verified_citations: list[VerifiedCitation],
    chunk_lookup: dict[str, dict[str, Any]],
    authoritative_domains: set[str],
) -> VerificationResult:
    """Roll citation outcomes up into a sentence-level tier."""
    if not verified_citations:
        return VerificationResult(
            tier=5,
            tier_label="hallucinated",
            mechanical_check="failed",
            semantic_check="skipped",
            failure_reason="Sentence has no verified citations.",
        )

    citation_results = [citation.verification for citation in verified_citations]

    if any(result.tier == 5 for result in citation_results):
        failure = next(
            (result.failure_reason for result in citation_results if result.tier == 5),
            "At least one citation failed mechanical verification.",
        )
        return VerificationResult(
            tier=5,
            tier_label="hallucinated",
            mechanical_check="failed",
            semantic_check="skipped",
            failure_reason=failure,
        )

    if any(result.tier == 4 for result in citation_results):
        failure = next(
            (result.failure_reason for result in citation_results if result.tier == 4),
            "At least one citation misrepresents its source.",
        )
        return VerificationResult(
            tier=4,
            tier_label="misrepresented",
            mechanical_check="passed",
            semantic_check="failed",
            failure_reason=failure,
        )

    all_semantic_passed = all(result.semantic_check == "passed" for result in citation_results)
    citation_domains = {
        str(chunk_lookup.get(citation.chunk_id, {}).get("domain", ""))
        for citation in verified_citations
        if chunk_lookup.get(citation.chunk_id, {}).get("domain")
    }
    authoritative_hit = any(
        is_authoritative_domain(domain, authoritative_domains) for domain in citation_domains
    )

    if all_semantic_passed and authoritative_hit:
        return VerificationResult(
            tier=1,
            tier_label="authoritative",
            mechanical_check="passed",
            semantic_check="passed",
            failure_reason=None,
        )

    if all_semantic_passed and len(citation_domains) >= 2:
        return VerificationResult(
            tier=2,
            tier_label="consensus",
            mechanical_check="passed",
            semantic_check="passed",
            failure_reason=None,
        )

    fallback_reason = next(
        (result.failure_reason for result in citation_results if result.failure_reason),
        None,
    )
    return VerificationResult(
        tier=3,
        tier_label="model_assisted",
        mechanical_check="passed",
        semantic_check="passed" if all_semantic_passed else "skipped",
        failure_reason=fallback_reason,
    )
def _verify_citation(
    claim_text: str,
    citation: Citation,
    chunk_lookup: dict[str, dict[str, Any]],
    model: str,
    authoritative: set[str],
) -> tuple[VerificationResult, str | None]:
    """
    Run semantic verification on one citation.

    Returns:
        (VerificationResult, rewrite_request_or_None)
    """
    chunk_id = citation.chunk_id
    chunk_data = chunk_lookup.get(chunk_id, {})
    domain = str(chunk_data.get("domain", ""))
    chunk_text = str(chunk_data.get("text", ""))
    source_metadata = json.dumps(
        {k: v for k, v in chunk_data.items() if k not in ("text", "chunk_id")},
        indent=2,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _USER_PROMPT_TEMPLATE.format(
                claim=claim_text,
                quote=citation.exact_source_quote,
                chunk_text=chunk_text or "(chunk text unavailable)",
                source_metadata=source_metadata or "{}",
            ),
        },
    ]

    try:
        completion_kwargs = build_completion_kwargs(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "semantic.llm_call",
            attributes={"model": model, "chunk_id": chunk_id},
        ):
            start = time.monotonic()
            with _llm_semaphore:
                response = litellm.completion(**completion_kwargs)
            LLM_CALL_DURATION.labels(node="semantic", model=model).observe(time.monotonic() - start)
        raw = response.choices[0].message.content or ""
        data = _parse_semantic_response(raw)
    except Exception as exc:
        logger.warning(
            "Semantic verification failed for chunk %s, degrading to Tier 3: %s",
            chunk_id,
            exc,
        )
        return (
            _degraded_verification(
                f"Semantic verifier error (degraded to deterministic fallback): {exc}",
            ),
            None,
        )

    if data["semantic_check"] == "failed":
        failure_reason = str(data["failure_reason"])
        return _failed_semantic_verification(failure_reason), failure_reason

    return _passed_verification(domain, authoritative), None


def semantic_verifier_node(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node — Semantic Verifier (Stage 2).

    Iterates over draft_sentences from state. For each mechanically-valid citation,
    runs the lightweight semantic check. Citation-level results are rolled up into
    a sentence-level verification summary.

    Returns keys: final_sentences, rewrite_requests, loop_count, audit_trail
    """
    audit: list[dict[str, Any]] = []

    pipeline_cfg: dict = state.get("pipeline_config") or {}
    stages_cfg: dict = pipeline_cfg.get("stages") or {}
    semantic_enabled: bool = stages_cfg.get("semantic_verification_enabled", True)

    models_cfg: dict = state.get("models_config") or {}
    model: str = models_cfg.get("verifier", "gpt-4o-mini")

    draft_sentences: list[dict] = list(state.get("draft_sentences") or [])
    indexed_chunks: list[dict] = list(state.get("indexed_chunks") or [])
    chunk_lookup: dict[str, dict[str, Any]] = {chunk["chunk_id"]: chunk for chunk in indexed_chunks}
    mechanical_results: dict[str, dict[str, Any]] = state.get("mechanical_results") or {}
    authoritative_domains, _ = build_domain_sets(state.get("app_config") or {})

    audit.append(
        _audit(
            "semantic_verifier_start",
            {
                "semantic_enabled": semantic_enabled,
                "model": model,
                "sentence_count": len(draft_sentences),
                "loop_count": state.get("loop_count", 0),
            },
        )
    )

    final_sentences: list[dict] = []
    rewrite_requests: list[str] = []

    for sentence_dict in draft_sentences:
        sentence_id = sentence_dict["sentence_id"]
        claim_text = sentence_dict["text"]
        citations = [Citation(**citation) for citation in sentence_dict.get("citations") or []]

        if not sentence_dict.get("is_cited") or not citations:
            rewrite_requests.append(_build_uncited_sentence_request(sentence_id))
            sentence_verification = VerificationResult(
                tier=5,
                tier_label="hallucinated",
                mechanical_check="failed",
                semantic_check="skipped",
                failure_reason="Sentence is unsupported because it has no citations.",
            )
            final_sentences.append(
                FinalSentence(
                    sentence_id=sentence_id,
                    text=claim_text,
                    is_cited=False,
                    citations=[],
                    verification=sentence_verification,
                ).model_dump()
            )
            audit.append(
                _audit(
                    "semantic_unsupported_sentence",
                    {"sentence_id": sentence_id, "reason": sentence_verification.failure_reason},
                )
            )
            continue

        verified_citations: list[VerifiedCitation] = []

        for citation in citations:
            mechanical_payload = mechanical_results.get(citation.citation_id)
            if mechanical_payload is None:
                vr = VerificationResult(
                    tier=5,
                    tier_label="hallucinated",
                    mechanical_check="failed",
                    semantic_check="skipped",
                    failure_reason="Citation was not processed by the mechanical verifier.",
                )
            elif isinstance(mechanical_payload, str):
                if mechanical_payload == "passed":
                    vr = VerificationResult(
                        tier=3,
                        tier_label="model_assisted",
                        mechanical_check="passed",
                        semantic_check="skipped",
                        failure_reason=None,
                    )
                else:
                    vr = VerificationResult(
                        tier=5,
                        tier_label="hallucinated",
                        mechanical_check="failed",
                        semantic_check="skipped",
                        failure_reason="Citation failed mechanical verification.",
                    )
            else:
                vr = VerificationResult.model_validate(mechanical_payload)

            if vr.mechanical_check == "passed":
                if not semantic_enabled:
                    vr = _degraded_verification(
                        "Semantic verification disabled; deterministic fallback applied.",
                    )
                    audit.append(
                        _audit(
                            "semantic_skipped_disabled",
                            {"citation_id": citation.citation_id, "chunk_id": citation.chunk_id},
                        )
                    )
                else:
                    vr, rewrite_reason = _verify_citation(
                        claim_text=claim_text,
                        citation=citation,
                        chunk_lookup=chunk_lookup,
                        model=model,
                        authoritative=authoritative_domains,
                    )
                    if rewrite_reason is not None:
                        rewrite_requests.append(
                            _build_tier4_rewrite_request(
                                sentence_id=sentence_id,
                                citation_id=citation.citation_id,
                                chunk_id=citation.chunk_id,
                                failure_reason=rewrite_reason,
                            )
                        )

            verified_citation = VerifiedCitation(
                citation_id=citation.citation_id,
                chunk_id=citation.chunk_id,
                exact_source_quote=citation.exact_source_quote,
                verification=vr,
            )
            verified_citations.append(verified_citation)

            audit.append(
                _audit(
                    "semantic_citation_result",
                    {
                        "citation_id": citation.citation_id,
                        "chunk_id": citation.chunk_id,
                        "tier": vr.tier,
                        "mechanical_check": vr.mechanical_check,
                        "semantic_check": vr.semantic_check,
                        "failure_reason": vr.failure_reason,
                    },
                )
            )

        sentence_verification = _aggregate_sentence_verification(
            verified_citations,
            chunk_lookup,
            authoritative_domains,
        )

        final_sentences.append(
            FinalSentence(
                sentence_id=sentence_id,
                text=claim_text,
                is_cited=True,
                citations=verified_citations,
                verification=sentence_verification,
            ).model_dump()
        )

    audit.append(
        _audit(
            "semantic_verifier_complete",
            {
                "final_sentence_count": len(final_sentences),
                "rewrite_request_count": len(rewrite_requests),
            },
        )
    )

    return {
        "final_sentences": final_sentences,
        "rewrite_requests": rewrite_requests,
        "loop_count": state.get("loop_count", 0) + 1,
        "audit_trail": audit,
    }
