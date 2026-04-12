"""
Axiom Engine v2.3 — FastAPI Gateway (Module 1)

Responsibilities:
  - Exposes POST /v1/synthesize as the single entry point.
  - Exposes GET /health for liveness/readiness probes.
  - Validates input via the AxiomRequest Pydantic model.
  - Converts AxiomRequest into a GraphState, invokes the compiled
    LangGraph DAG, and marshals the result into AxiomResponse.
  - Computes the ConfidenceSummary (tier breakdown + overall score).
  - Catches all unhandled exceptions and returns a structured error
    response matching the AxiomResponse schema (architecture §7).
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.metadata
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Literal

from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from axiom_engine.config.logging import configure_logging, request_id_ctx
from axiom_engine.config.observability import (
    CACHE_HITS,
    CACHE_MISSES,
    PIPELINE_DURATION,
    REQUESTS_BY_STATUS,
    TIER_ASSIGNMENTS,
    run_with_otel_context,
    setup_prometheus,
    setup_tracing,
)
from axiom_engine.graph import build_axiom_graph
from axiom_engine.models import (
    AuditEvent,
    AxiomRequest,
    AxiomResponse,
    ConfidenceSummary,
    DebugInfo,
    FinalSentence,
    TierBreakdown,
)
from axiom_engine.nodes.retriever import set_search_backend
from axiom_engine.state import make_initial_state

load_dotenv()  # Load .env from project root; no-op if absent.

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("axiom_engine")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

# Configurable via env var; default is 20 requests/minute per IP.
# Set AXIOM_RATE_LIMIT=100/minute in production if needed.
_RATE_LIMIT = os.environ.get("AXIOM_RATE_LIMIT", "20/minute")

limiter = Limiter(key_func=get_remote_address, default_limits=[_RATE_LIMIT])


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_NON_PROD_ENVS = {"development", "dev", "local", "test"}


def _app_env() -> str:
    """Return the current runtime environment."""
    return os.environ.get("AXIOM_ENV", "development").strip().lower()


def _api_keys() -> set[str]:
    """Read valid API keys from the current environment."""
    return {k.strip() for k in os.environ.get("AXIOM_API_KEYS", "").split(",") if k.strip()}


def _auth_required() -> bool:
    """Require auth outside explicitly non-production environments."""
    return _app_env() not in _NON_PROD_ENVS


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str | None:
    """Validate the API key if authentication is enabled."""
    valid_keys = _api_keys()
    if not valid_keys:
        if _auth_required():
            raise HTTPException(status_code=503, detail="Server authentication is misconfigured.")
        return None
    if not api_key or api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return api_key


# ---------------------------------------------------------------------------
# Response cache (bounded, TTL-based)
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS: int = 300  # 5 minutes
_CACHE_MAX_SIZE: int = 256  # max cached responses before LRU eviction
_response_cache: TTLCache[str, dict[str, Any]] = TTLCache(
    maxsize=_CACHE_MAX_SIZE, ttl=_CACHE_TTL_SECONDS
)
_cache_lock = threading.Lock()


def _cache_key(payload: AxiomRequest) -> str:
    """SHA-256 of the request fields that shape the response body."""
    raw = json.dumps(
        {
            "query": payload.user_query,
            "models": payload.models.model_dump(),
            "pipeline": payload.pipeline_config.model_dump(),
            "app": payload.app_config.model_dump(),
            "include_debug": payload.include_debug,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _response_to_cache_value(response: AxiomResponse) -> dict[str, Any]:
    """Serialize a response without request-scoped identifiers for safe reuse."""
    data = response.model_dump()
    data.pop("request_id", None)
    return data


def _hydrate_cached_response(request_id: str, cached: dict[str, Any]) -> AxiomResponse:
    """Rebuild a response for the current request from cached template data."""
    return AxiomResponse.model_validate({"request_id": request_id, **copy.deepcopy(cached)})


def _get_cached(key: str, request_id: str) -> AxiomResponse | None:
    with _cache_lock:
        cached = _response_cache.get(key)
    if cached is None:
        return None
    return _hydrate_cached_response(request_id, cached)


def _set_cached(key: str, response: AxiomResponse) -> None:
    with _cache_lock:
        _response_cache[key] = _response_to_cache_value(response)


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
) -> Literal["success", "partial", "unanswerable", "error"]:
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

    if not final_sentences:
        return "partial"

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
    include_debug: bool = False,
) -> AxiomResponse:
    """
    Convert the raw GraphState dict returned by the compiled graph into
    a validated AxiomResponse.
    """
    is_answerable: bool = graph_result.get("is_answerable", False)
    raw_sentences: list[dict] = graph_result.get("final_sentences", [])

    # Validate each sentence through the Pydantic model to ensure
    # the response contract is fully honoured.
    final_sentences: list[FinalSentence] = [FinalSentence.model_validate(s) for s in raw_sentences]

    status = determine_status(is_answerable, raw_sentences)
    confidence = compute_confidence_summary(raw_sentences)

    debug: DebugInfo | None = None
    if include_debug:
        raw_audit = graph_result.get("audit_trail", [])
        debug = DebugInfo(
            audit_trail=[AuditEvent.model_validate(e) for e in raw_audit],
            pipeline_stats={
                "chunks_retrieved": len(graph_result.get("indexed_chunks", [])),
                "chunks_ranked": len(graph_result.get("ranked_chunks", [])),
                "loop_count": graph_result.get("loop_count", 0),
                "retrieval_retry_count": graph_result.get("retrieval_retry_count", 0),
            },
        )

    return AxiomResponse(
        request_id=request_id,
        status=status,
        is_answerable=is_answerable,
        confidence_summary=confidence,
        final_response=final_sentences,
        debug=debug,
    )


def make_error_response(
    request_id: str,
    error: Exception,
) -> AxiomResponse:
    """
    Build a structured error response matching the AxiomResponse schema.
    Category 1 errors (architecture §7): unrecoverable system failures.
    """
    # Log full detail server-side; return only a generic message to the client.
    logger.error(
        "Pipeline error for request %s: %s: %s",
        request_id,
        type(error).__name__,
        error,
    )

    return AxiomResponse(
        request_id=request_id,
        status="error",
        is_answerable=False,
        confidence_summary=ConfidenceSummary(
            overall_score=0.0,
            tier_breakdown=TierBreakdown(),
        ),
        final_response=[],
        error_message=f"Internal pipeline error — see server logs for request {request_id}.",
    )


# ---------------------------------------------------------------------------
# App factory & lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Compile the graph once at startup, attach to app state."""
    configure_logging()

    if _auth_required() and not _api_keys():
        raise RuntimeError("AXIOM_API_KEYS must be configured when AXIOM_ENV is not development.")

    # Wire search backend — Tavily if key present, else MockSearchBackend.
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        from axiom_engine.search.tavily import TavilySearchBackend

        set_search_backend(TavilySearchBackend(api_key=tavily_key))
        logger.info("Search backend: Tavily (live web search enabled).")
    else:
        logger.warning(
            "TAVILY_API_KEY not set — using MockSearchBackend. "
            "Set TAVILY_API_KEY in .env for live web search."
        )

    setup_tracing(app, "axiom-engine", _VERSION)

    app.state.engine = build_axiom_graph()
    logger.info("Axiom Engine graph compiled and ready.")
    yield
    logger.info("Axiom Engine shutting down.")


_VERSION = importlib.metadata.version("axiom-engine")

# Disable interactive API docs in production (AXIOM_DOCS_ENABLED=false).
_docs_enabled = os.environ.get("AXIOM_DOCS_ENABLED", "true").lower() == "true"

app = FastAPI(
    title="Axiom Engine",
    version=_VERSION,
    description="Configuration-driven Agentic RAG with 6-tier verification.",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
)

# CORS — locked down by default; set AXIOM_CORS_ORIGINS to allow specific origins.
_cors_origins = [
    o.strip() for o in os.environ.get("AXIOM_CORS_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# Prometheus instrumentation must happen at module level (before the app starts),
# because it adds middleware which Starlette forbids after startup.
setup_prometheus(app)


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
        logger.debug("Could not parse request body for error context")

    logger.exception("Unhandled exception for request %s", request_id)
    error_response = make_error_response(request_id, exc)
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(),
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", summary="Liveness / readiness probe.")
@limiter.exempt
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post(
    "/v1/synthesize",
    response_model=AxiomResponse,
    summary="Run the Axiom Engine verification pipeline.",
)
async def synthesize(
    payload: AxiomRequest,
    _api_key: str | None = Depends(verify_api_key),
) -> AxiomResponse:
    """
    Accept an AxiomRequest, execute the LangGraph DAG, and return
    a fully validated AxiomResponse with tier breakdown and confidence score.
    """
    request_id_ctx.set(payload.request_id)

    # Cache lookup — skip for error/unanswerable; only cache clean results.
    key = _cache_key(payload)
    cached = _get_cached(key, payload.request_id)
    if cached is not None:
        CACHE_HITS.inc()
        logger.info("Cache hit for request %s", payload.request_id)
        REQUESTS_BY_STATUS.labels(status=cached.status).inc()
        return cached
    CACHE_MISSES.inc()

    initial_state = make_initial_state(
        request_id=payload.request_id,
        user_query=payload.user_query,
        app_config=payload.app_config.model_dump(),
        models_config=payload.models.model_dump(),
        pipeline_config=payload.pipeline_config.model_dump(),
    )

    try:
        engine = app.state.engine
        ctx_fn = run_with_otel_context(engine.invoke, initial_state)
        with PIPELINE_DURATION.time():
            graph_result = await asyncio.to_thread(ctx_fn)
    except Exception as exc:
        REQUESTS_BY_STATUS.labels(status="error").inc()
        return make_error_response(payload.request_id, exc)

    response = marshal_response(payload.request_id, graph_result, payload.include_debug)

    # Update Prometheus metrics.
    REQUESTS_BY_STATUS.labels(status=response.status).inc()
    for sentence in graph_result.get("final_sentences", []):
        tier = sentence.get("verification", {}).get("tier", 3)
        TIER_ASSIGNMENTS.labels(tier=str(tier)).inc()

    # Cache successful and partial responses.
    if response.status in ("success", "partial"):
        _set_cached(key, response)

    return response
