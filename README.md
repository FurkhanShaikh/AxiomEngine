# Axiom Engine

**Citation-verified RAG with 6-tier confidence scoring.**

Axiom Engine is a retrieval-augmented generation (RAG) service that
verifies every cited claim before presenting answers. Each claim is assigned
a confidence tier (1-6) based on deterministic + semantic verification.

## Install

```bash
pip install axiom-rag
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add axiom-rag
```

## Quick start

### From PyPI

```bash
# Set required env vars (or create a .env file)
export AXIOM_ENV=development
export TAVILY_API_KEY=your_key   # or use AXIOM_ALLOW_MOCK_SEARCH=true

# Start the server
axiom-engine serve

# In another terminal — send a test query
axiom-engine probe "What are solid-state batteries?"

# Check resolved configuration (secrets redacted)
axiom-engine check-config
```

### From source

```bash
git clone https://github.com/FurkhanShaikh/axiom-engine.git
cd axiom-engine
python tasks.py install          # scaffold .env + install deps via uv
# Edit .env — fill in TAVILY_API_KEY for live web search
python tasks.py run              # start FastAPI server at http://localhost:8000
python tasks.py probe "your question"
```

## Configuration

All settings are controlled via environment variables (or a `.env` file).
No code changes required. Run `axiom-engine check-config` to see the full
resolved configuration.

| Variable | Default | Description |
|---|---|---|
| `AXIOM_ENV` | `production` | Runtime environment. Set to `development` to disable auth. |
| `AXIOM_API_KEYS` | _(empty)_ | Comma-separated API keys. Required when env != development. |
| `TAVILY_API_KEY` | _(empty)_ | Tavily search API key for live web retrieval. |
| `AXIOM_DEFAULT_SYNTHESIZER_MODEL` | `claude-sonnet-4-5` | LiteLLM model ID for synthesis. |
| `AXIOM_DEFAULT_VERIFIER_MODEL` | `gpt-4o-mini` | LiteLLM model ID for semantic verification. |
| `AXIOM_RATE_LIMIT` | `20/minute` | Rate limit per API key or IP. |
| `AXIOM_CACHE_TTL_SECONDS` | `300` | Response cache TTL. |
| `AXIOM_REDIS_URL` | _(empty)_ | Optional Redis URL for distributed cache. |
| `AXIOM_CORS_ORIGINS` | _(empty)_ | Comma-separated allowed CORS origins. |
| `AXIOM_DOCS_ENABLED` | `true` | Set `false` to disable /docs and /redoc. |
| `AXIOM_SEMANTIC_VERIFICATION_ENABLED` | `true` | Enable/disable Stage 2 semantic verification. |
| `LOG_FORMAT` | `text` | `json` for structured log output. |

See [.env.example](.env.example) for the full list with comments.

## Architecture

```
retriever -> scorer -> ranker -> synthesizer -> verifier -+
   ^                    ^                                 |
   |                    +-- (rewrite loop) <--------------+  (Tier 4/5 & loop < max)
   +-- (re-retrieve) <-----------------------------------+  (loop exhausted & retries left)
```

| Module | Responsibility |
|---|---|
| **Retriever** | Web search via Tavily, dedup, HTML strip, paragraph chunking |
| **Scorer** | Domain authority + content quality scoring (deterministic) |
| **Ranker** | BM25-based relevance ranking with quality blend |
| **Synthesizer** | LLM-powered answer generation with strict citation format |
| **Verifier** | Two-stage verification: mechanical (exact match) + semantic (LLM) |

## Verification tiers

| Tier | Label | Meaning |
|---|---|---|
| 1 | Authoritative | Verified against official/primary source |
| 2 | Multi-Source | Verified against multiple independent domains |
| 3 | Model Assisted | Mechanically verified; semantic relied on model knowledge |
| 4 | Misrepresented | Quote exists but claim distorts context |
| 5 | Hallucinated | Quote not found in source chunk |
| 6 | Conflicted | Reserved for future contradiction detection |

## CLI reference

```bash
axiom-engine serve [--host 0.0.0.0] [--port 8000] [--reload]
axiom-engine probe "question" [--url URL] [--model MODEL] [--debug]
axiom-engine check-config [--format text|json]
```

## Development

```bash
python tasks.py test             # unit tests (>=70% coverage required)
python tasks.py lint             # ruff + mypy
python tasks.py format           # auto-format
python tasks.py clean            # remove caches + venv
```

## API

- `POST /v1/synthesize` — Run the verification pipeline
- `GET /health` — Liveness probe
- `GET /health/ready` — Readiness probe
- `GET /metrics` — Prometheus metrics

See the interactive docs at `http://localhost:8000/docs` when the server is running.

## Docker

```bash
docker compose up --build
```

## License

[MIT](LICENSE)
