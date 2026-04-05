"""
Axiom Engine — Structured logging configuration.

Call configure_logging() once at application startup (e.g. in the FastAPI
lifespan) to set up consistent, machine-readable log output.

Supports two output formats (controlled via LOG_FORMAT env var):
  - "text"  : Human-readable, coloured output for local development (default).
  - "json"  : Machine-readable JSON lines for production log aggregation.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

# ---------------------------------------------------------------------------
# Context variable for request-scoped correlation
# ---------------------------------------------------------------------------

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class _TextFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.FORMAT, datefmt="%Y-%m-%dT%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_ctx.get()
        if rid:
            record.msg = f"[{rid}] {record.msg}"
        return super().format(record)


class _JSONFormatter(logging.Formatter):
    """Structured JSON formatter for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = request_id_ctx.get()
        if rid:
            entry["request_id"] = rid
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure the root 'axiom_engine' logger.

    Set LOG_FORMAT=json (env var) for structured JSON output, otherwise
    defaults to human-readable text.
    """
    root_logger = logging.getLogger("axiom_engine")
    if root_logger.handlers:
        return  # Already configured (avoid duplicate handlers on reload)

    root_logger.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    fmt = os.environ.get("LOG_FORMAT", "text").lower()
    if fmt == "json":
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(_TextFormatter())

    root_logger.addHandler(handler)
