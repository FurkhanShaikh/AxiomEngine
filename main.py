"""
Axiom Engine v2.3 — FastAPI Gateway (Module 1)

Responsibilities:
  - Exposes POST /v1/synthesize as the single entry point.
  - Validates input via the AxiomRequest Pydantic model.
  - Converts AxiomRequest into a GraphState, invokes the compiled
    LangGraph DAG, and marshals the result into AxiomResponse.
  - Computes the ConfidenceSummary (tier breakdown + overall score).
  - Catches all unhandled exceptions and returns a structured error
    response matching the AxiomResponse schema (architecture §7).
"""

from __future__ import annotations

import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from graph import build_axiom_graph
from models import (
    AxiomRequest,
    AxiomResponse,
    ConfidenceSummary,
    FinalSentence,
    TierBreakdown,
)
from state import make_initial_state


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

# Tier weights for the overall confidence score (architecture §4):
#   Tier 1 (Authoritative)    → 1.0
#   Tier 2 (Consensus)        → 0.85
#   Tier 3 (Model Assisted)   → 0.60
#   Tier 4 (Misrepresented)   → 0.20  (should rarely survive to final output)
#   Tier 5 (Hallucinated)     → 0.00  (should never survive to final output)
#   Tier 6 (Conflicted)       → 0.40
_TIER_WEIGHTS: dict[int, float] = {
    1: 1.0,
    2: 0.85,
    3: 0.60,
    4: 0.20,
    5: 0.00,
    6: 0.40,
}


def compute_confidence_summary(
    final_sentences: list[dict[str, Any]],
) -> ConfidenceSummary:
    """
    Compute tier breakdown and weighted overall confidence score from
    the verified final_sentences produced by the graph.
    """
    breakdown = TierBreakdown()
    weighted_sum = 0.0
    total_claims = 0

    for sentence in final_sentences:
        vr = sentence.get("verification", {})
        tier: int = vr.get("tier", 3)

        attr = f"tier_{tier}_claims"
        setattr(breakdown, attr, getattr(breakdown, attr, 0) + 1)

        weighted_sum += _TIER_WEIGHTS.get(tier, 0.0)
        total_claims += 1

    overall = round(weighted_sum / total_claims, 4) if total_claims > 0 else 0.0

    return ConfidenceSummary(
        overall_score=overall,
        tier_breakdown=breakdown,
    )


def determine_status(
    is_answerable: bool,
    final_sentences: list[dict[str, Any]],
) -> str:
    """
    Determine the response status string.

    Rules:
      - "unanswerable" if escape hatch fired.
      - "success" if all sentences are Tier 1–3.
      - "partial" if any sentence is Tier 4, 5, or 6.
      - "error" should only come from exception handling (not here).
    """
    if not is_answerable:
        return "unanswerable"

    for s in final_sentences:
        tier = s.get("verification", {}).get("tier", 3)
        if tier in (4, 5, 6):
            return "partial"

    return "success"


# ---------------------------------------------------------------------------
# Graph result → AxiomResponse marshalling
# ---------------------------------------------------------------------------

def marshal_response(
    request_id: str,
    graph_result: dict[str, Any],
) -> AxiomResponse:
    """
    Convert the raw GraphState dict returned by the compiled graph into
    a validated AxiomResponse.
    """
    is_answerable: bool = graph_result.get("is_answerable", False)
    raw_sentences: list[dict] = graph_result.get("final_sentences", [])

    # Validate each sentence through the Pydantic model to ensure
    # the response contract is fully honoured.
    final_sentences: list[FinalSentence] = [
        FinalSentence.model_validate(s) for s in raw_sentences
    ]

    status = determine_status(is_answerable, raw_sentences)
    confidence = compute_confidence_summary(raw_sentences)

    return AxiomResponse(
        request_id=request_id,
        status=status,
        is_answerable=is_answerable,
        confidence_summary=confidence,
        final_response=final_sentences,
    )


def make_error_response(
    request_id: str,
    error: Exception,
) -> AxiomResponse:
    """
    Build a structured error response matching the AxiomResponse schema.
    Category 1 errors (architecture §7): unrecoverable system failures.
    """
    return AxiomResponse(
        request_id=request_id,
        status="error",
        is_answerable=False,
        confidence_summary=ConfidenceSummary(
            overall_score=0.0,
            tier_breakdown=TierBreakdown(),
        ),
        final_response=[],
    )


# ---------------------------------------------------------------------------
# App factory & lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Compile the graph once at startup, attach to app state."""
    app.state.engine = build_axiom_graph()
    yield


app = FastAPI(
    title="Axiom Engine",
    version="2.3.0",
    description="Configuration-driven Agentic RAG with 6-tier verification.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Global exception handler (architecture §7, Category 1)
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Catch-all for any unhandled exception that escapes the endpoint.
    Returns a structured AxiomResponse with status="error".
    """
    # Try to extract request_id from the body; fall back to "unknown".
    request_id = "unknown"
    try:
        body = await request.json()
        request_id = body.get("request_id", "unknown")
    except Exception:
        pass

    error_response = make_error_response(request_id, exc)
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(),
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/v1/synthesize",
    response_model=AxiomResponse,
    summary="Run the Axiom Engine verification pipeline.",
)
async def synthesize(payload: AxiomRequest) -> AxiomResponse:
    """
    Accept an AxiomRequest, execute the LangGraph DAG, and return
    a fully validated AxiomResponse with tier breakdown and confidence score.
    """
    initial_state = make_initial_state(
        request_id=payload.request_id,
        user_query=payload.user_query,
        app_config=payload.app_config.model_dump(),
        models_config=payload.models.model_dump(),
        pipeline_config=payload.pipeline_config.model_dump(),
    )

    try:
        engine = app.state.engine
        graph_result = engine.invoke(initial_state)
    except Exception as exc:
        return make_error_response(payload.request_id, exc)

    return marshal_response(payload.request_id, graph_result)
