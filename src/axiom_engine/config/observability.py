"""
Axiom Engine — Observability setup (Prometheus metrics + OpenTelemetry tracing).

Call setup_prometheus() and setup_tracing() once at startup in the FastAPI lifespan.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from fastapi import FastAPI
from opentelemetry import context, trace
from opentelemetry.trace import Tracer
from prometheus_client import Counter, Histogram

logger = logging.getLogger("axiom_engine.observability")

# ---------------------------------------------------------------------------
# Prometheus — custom domain metrics
# ---------------------------------------------------------------------------

PIPELINE_DURATION = Histogram(
    "axiom_pipeline_duration_seconds",
    "End-to-end pipeline duration per request",
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120],
)

LLM_CALL_DURATION = Histogram(
    "axiom_llm_call_duration_seconds",
    "Wall-clock duration of a single LLM completion call",
    ["node", "model"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60],
)

CACHE_HITS = Counter("axiom_cache_hits_total", "Response cache hits")
CACHE_MISSES = Counter("axiom_cache_misses_total", "Response cache misses")
REQUESTS_BY_STATUS = Counter(
    "axiom_requests_by_status_total",
    "Request outcomes by status",
    ["status"],
)
TIER_ASSIGNMENTS = Counter(
    "axiom_tier_assignments_total",
    "Verification tier assignment count",
    ["tier"],
)

_prometheus_initialized = False


def setup_prometheus(app: FastAPI) -> None:
    """Instrument the FastAPI app with Prometheus metrics (idempotent)."""
    global _prometheus_initialized
    if _prometheus_initialized:
        return

    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, include_in_schema=False)
    _prometheus_initialized = True
    logger.info("Prometheus metrics enabled at /metrics.")


# ---------------------------------------------------------------------------
# OpenTelemetry — distributed tracing
# ---------------------------------------------------------------------------

_tracer: Tracer = trace.get_tracer("axiom-engine")


def setup_tracing(app: FastAPI, service_name: str, version: str) -> None:
    """
    Configure OpenTelemetry tracing if OTEL_EXPORTER_OTLP_ENDPOINT is set.

    When the endpoint is not configured, the tracer remains a no-op (zero overhead).
    """
    global _tracer

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set — tracing disabled (no-op).")
        return

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": service_name, "service.version": version})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)

    _tracer = trace.get_tracer(service_name, version)
    logger.info("OpenTelemetry tracing enabled → %s", endpoint)


def get_tracer() -> Tracer:
    """Return the configured tracer (no-op when tracing is disabled)."""
    return _tracer


# ---------------------------------------------------------------------------
# Thread context propagation helper
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def run_with_otel_context(fn: Callable[..., Any], *args: Any) -> Callable[[], Any]:
    """
    Capture the current OTel context and return a zero-arg callable that
    reattaches it before invoking fn(*args).

    Use with asyncio.to_thread() to propagate trace context across the
    async → sync thread boundary:

        ctx_fn = run_with_otel_context(engine.invoke, initial_state)
        result = await asyncio.to_thread(ctx_fn)
    """
    ctx = context.get_current()

    @wraps(fn)
    def _wrapper() -> Any:
        token = context.attach(ctx)
        try:
            return fn(*args)
        finally:
            context.detach(token)

    return _wrapper
