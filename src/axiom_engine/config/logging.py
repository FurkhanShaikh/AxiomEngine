"""
Axiom Engine — Structured logging configuration.

Call configure_logging() once at application startup (e.g. in the FastAPI
lifespan) to set up consistent, machine-readable log output.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure the root 'axiom_engine' logger with a human-readable
    format directed to stderr.
    """
    root_logger = logging.getLogger("axiom_engine")
    if root_logger.handlers:
        return  # Already configured (avoid duplicate handlers on reload)

    root_logger.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
