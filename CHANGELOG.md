# Changelog

All notable changes to Axiom Engine are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0b1] - 2026-04-15

First public beta release.

### Added
- **RAG pipeline** — LangGraph DAG with retriever, scorer, ranker, synthesizer, and two-stage verifier (mechanical + semantic).
- **6-tier confidence scoring** — every cited claim is assigned a verification tier (1-Authoritative through 6-Conflicted).
- **Central configuration** (`config/settings.py`) — all `AXIOM_*` env vars in one typed `Settings` class backed by `pydantic-settings`. No code changes needed to configure.
- **CLI entry point** (`axiom-engine`) with `serve`, `probe`, and `check-config` subcommands.
- **FastAPI HTTP API** — `POST /v1/synthesize`, health probes, Prometheus metrics.
- **Search backends** — Tavily live web search with automatic fallback to mock backend.
- **LLM flexibility** — any LiteLLM-supported model, including local Ollama.
- **Response cache** — in-memory TTLCache with optional Redis backing layer.
- **Security hardening** — fail-closed auth, CORS lockdown, SSRF defense, rate limiting, body-size cap.
- **Observability** — Prometheus metrics, OpenTelemetry tracing, structured JSON logging.
- **CI pipeline** — GitHub Actions for lint, typecheck, test (3.11/3.12/3.13), security audit, Docker build.
- **Publish workflow** — tag-triggered release to TestPyPI (rc tags) and PyPI via Trusted Publishing.
- `tasks.py` developer task runner (install, run, test, lint, format, probe, clean).

[0.1.0b1]: https://github.com/FurkhanShaikh/axiom-engine/releases/tag/v0.1.0b1
