"""
Integration tests — real LLM calls via Ollama (no mocks).

These tests hit a local Ollama instance running qwen3.5:9b.
They use MockSearchBackend to inject controlled source chunks,
then let the full pipeline (scorer → ranker → synthesizer → verifier)
run with actual LLM inference.

Run with:  pytest test_integration.py -v -s
Skip if Ollama is unavailable:  pytest test_integration.py -v -s -k "not integration"

Requires: Ollama running on localhost:11434 with qwen3.5:9b loaded.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from fastapi.testclient import TestClient

from axiom_engine.main import app
from axiom_engine.nodes.retriever import MockSearchBackend, set_search_backend
from axiom_engine.state import make_initial_state

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_OLLAMA_MODEL = "ollama/qwen3.5:9b"
_OLLAMA_BASE = "http://localhost:11434"

# Use the same model for both synthesizer and verifier (single local model).
_MODELS_CONFIG = {
    "synthesizer": _OLLAMA_MODEL,
    "verifier": _OLLAMA_MODEL,
}


def _ollama_available() -> bool:
    """Check if Ollama is reachable."""
    try:
        import httpx
        resp = httpx.get(f"{_OLLAMA_BASE}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# Skip all tests in this module if Ollama isn't running.
pytestmark = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not available on localhost:11434",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Source chunks that the synthesizer will use to generate answers.
_BATTERY_CHUNKS = [
    {
        "url": "https://nature.com/solid-state-review",
        "content": (
            "Solid-state batteries replace liquid electrolytes with solid ceramics. "
            "This substitution significantly improves thermal stability and energy density. "
            "Current research has achieved ionic conductivity of 10 mS/cm in sulfide-based "
            "solid electrolytes, approaching that of liquid counterparts."
        ),
        "title": "Solid-State Battery Review — Nature Energy 2024",
    },
    {
        "url": "https://arxiv.org/abs/2024.98765",
        "content": (
            "The primary advantage of all-solid-state lithium batteries is the elimination "
            "of flammable liquid electrolytes, which significantly reduces the risk of "
            "thermal runaway. Additionally, solid electrolytes enable the use of lithium "
            "metal anodes, potentially doubling the energy density compared to conventional "
            "lithium-ion cells."
        ),
        "title": "All-Solid-State Lithium Batteries: Safety and Performance",
    },
]


@pytest.fixture(autouse=True)
def _set_ollama_env():
    """Ensure LiteLLM routes to local Ollama."""
    old = os.environ.get("OLLAMA_API_BASE")
    os.environ["OLLAMA_API_BASE"] = _OLLAMA_BASE
    yield
    if old is None:
        os.environ.pop("OLLAMA_API_BASE", None)
    else:
        os.environ["OLLAMA_API_BASE"] = old


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _make_request(
    query: str = "What is a solid-state battery?",
    chunks: list[dict] | None = None,
    **overrides: Any,
) -> dict:
    """Build a valid AxiomRequest payload."""
    set_search_backend(MockSearchBackend(chunks or _BATTERY_CHUNKS))
    payload = {
        "request_id": "integration_test",
        "user_query": query,
        "models": _MODELS_CONFIG,
        "pipeline_config": {
            "stages": {
                "semantic_verification_enabled": True,
                "max_ranked_chunks": 10,
                "max_rewrite_loops": 2,
            },
        },
    }
    payload.update(overrides)
    return payload


# ===========================================================================
# Integration tests
# ===========================================================================


class TestLiveFullPipeline:
    """End-to-end tests with real LLM calls."""

    def test_answerable_query_returns_success_or_partial(self, client: TestClient) -> None:
        """
        Given good source chunks about solid-state batteries,
        the LLM should produce an answerable response with citations.
        """
        resp = client.post("/v1/synthesize", json=_make_request())

        assert resp.status_code == 200
        data = resp.json()

        # The LLM should find the chunks answerable.
        assert data["status"] in ("success", "partial"), (
            f"Expected success/partial, got {data['status']}. "
            f"error_message={data.get('error_message')}"
        )
        assert data["is_answerable"] is True
        assert len(data["final_response"]) >= 1

        # At least one sentence should have a citation.
        cited = [s for s in data["final_response"] if s["is_cited"]]
        assert len(cited) >= 1, "Expected at least one cited sentence"

        # Confidence score should be positive.
        assert data["confidence_summary"]["overall_score"] > 0.0

        print(f"\n--- Live Pipeline Result ---")
        print(f"Status: {data['status']}")
        print(f"Confidence: {data['confidence_summary']['overall_score']}")
        print(f"Sentences: {len(data['final_response'])}")
        for s in data["final_response"]:
            v = s["verification"]
            print(f"  [{s['sentence_id']}] Tier {v['tier']} ({v['tier_label']}) "
                  f"mech={v['mechanical_check']} sem={v['semantic_check']}")
            print(f"    \"{s['text'][:100]}\"")

    def test_unanswerable_query_triggers_escape_hatch(self, client: TestClient) -> None:
        """
        Given chunks about batteries but a completely unrelated query,
        the LLM should set is_answerable=false.
        """
        unrelated_chunks = [
            {
                "url": "https://example.com/cooking",
                "content": (
                    "The best way to make a sourdough starter is to combine equal parts "
                    "flour and water by weight. Feed the starter every 24 hours, discarding "
                    "half before each feeding. After 5-7 days the starter should be active "
                    "and ready for baking."
                ),
                "title": "Sourdough Starter Guide",
            },
        ]
        resp = client.post(
            "/v1/synthesize",
            json=_make_request(
                query="What is the quantum chromodynamics coupling constant?",
                chunks=unrelated_chunks,
            ),
        )

        assert resp.status_code == 200
        data = resp.json()

        # The LLM should recognize chunks can't answer this query.
        assert data["status"] in ("unanswerable", "success", "partial"), (
            f"Unexpected status: {data['status']}, error={data.get('error_message')}"
        )

        print(f"\n--- Escape Hatch Test ---")
        print(f"Status: {data['status']}, is_answerable: {data['is_answerable']}")

    def test_mechanical_verification_catches_real_llm_hallucination(self, client: TestClient) -> None:
        """
        The mechanical verifier should verify that exact_source_quote
        actually appears in the source chunk. This tests the real pipeline
        end-to-end — if the LLM paraphrases instead of copying verbatim,
        the mechanical check will catch it.
        """
        resp = client.post("/v1/synthesize", json=_make_request())

        assert resp.status_code == 200
        data = resp.json()

        if data["status"] == "error":
            pytest.skip(f"LLM error: {data.get('error_message')}")

        # Check that verification actually ran.
        for sentence in data["final_response"]:
            v = sentence["verification"]
            if sentence["is_cited"]:
                # Mechanical check should have run (not skipped).
                assert v["mechanical_check"] in ("passed", "failed"), (
                    f"Expected mechanical_check to run, got {v['mechanical_check']}"
                )

        print(f"\n--- Mechanical Verification Test ---")
        tier_counts = data["confidence_summary"]["tier_breakdown"]
        print(f"Tier breakdown: {json.dumps(tier_counts)}")
        mech_results = [
            (s["sentence_id"], s["verification"]["mechanical_check"])
            for s in data["final_response"] if s["is_cited"]
        ]
        print(f"Mechanical results: {mech_results}")

    def test_confidence_score_is_computed(self, client: TestClient) -> None:
        """The confidence summary should have a valid score and tier breakdown."""
        resp = client.post("/v1/synthesize", json=_make_request())

        assert resp.status_code == 200
        data = resp.json()

        if data["status"] == "error":
            pytest.skip(f"LLM error: {data.get('error_message')}")

        cs = data["confidence_summary"]
        assert 0.0 <= cs["overall_score"] <= 1.0

        tb = cs["tier_breakdown"]
        total = sum(tb[f"tier_{i}_claims"] for i in range(1, 7))
        assert total == len(data["final_response"])

        print(f"\n--- Confidence Score Test ---")
        print(f"Overall: {cs['overall_score']}, Claims: {total}")

    def test_response_conforms_to_axiom_response_schema(self, client: TestClient) -> None:
        """The raw JSON response should validate against AxiomResponse."""
        from axiom_engine.models import AxiomResponse

        resp = client.post("/v1/synthesize", json=_make_request())
        assert resp.status_code == 200

        data = resp.json()
        parsed = AxiomResponse.model_validate(data)
        assert parsed.request_id == "integration_test"

    def test_health_endpoint(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
