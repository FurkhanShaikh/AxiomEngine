"""Cross-source contradiction detection → Tier 6 (opt-in).

When AXIOM_CONTRADICTION_DETECTION_ENABLED is on, a multi-domain sentence whose
cited sources actively contradict each other is surfaced as Tier 6 (Conflicted)
instead of a confident Tier 1/2. These tests mock the verifier LLM so they need
no keys, routing the call types (per-citation semantic check vs. corroboration
vs. contradiction) by their system prompt.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from axiom_rag_engine.config.settings import get_settings
from axiom_rag_engine.nodes.semantic import (
    _parse_contradiction_response,
    semantic_verifier_node,
)
from axiom_rag_engine.state import make_initial_state


class TestParseContradictionResponse:
    def test_parses_true(self) -> None:
        ok, reason = _parse_contradiction_response(
            '{"contradicted": true, "reasoning": "opposite conclusions"}'
        )
        assert ok is True
        assert reason == "opposite conclusions"

    def test_parses_false(self) -> None:
        ok, _ = _parse_contradiction_response('{"contradicted": false, "reasoning": "same fact"}')
        assert ok is False

    def test_strips_fences_and_think(self) -> None:
        raw = '<think>hmm</think>```json\n{"contradicted": true, "reasoning": "x"}\n```'
        ok, _ = _parse_contradiction_response(raw)
        assert ok is True

    def test_rejects_non_bool(self) -> None:
        with pytest.raises(ValueError, match="contradicted must be a bool"):
            _parse_contradiction_response('{"contradicted": "yes"}')

    def test_rejects_malformed_json(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            _parse_contradiction_response("not json at all")


# ---------------------------------------------------------------------------
# Integration: the Tier 6 gate
# ---------------------------------------------------------------------------

_SEMANTIC_PASS = json.dumps(
    {"semantic_check": "passed", "failure_reason": None, "reasoning": "faithful"}
)


def _resp(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    response.usage = None
    return response


def _router(contradiction_json: str, corroboration_json: str = '{"corroborated": true}') -> Any:
    """Route by system prompt: contradiction vs. corroboration vs. semantic check."""

    async def _route(**kwargs: Any) -> MagicMock:
        # The semantic-verification prompt also mentions "contradiction tiers", so
        # route on a phrase unique to the contradiction gate's prompt.
        system = kwargs["messages"][0]["content"].lower()
        if "contradict each other" in system:
            return _resp(contradiction_json)
        if "corroborate" in system:
            return _resp(corroboration_json)
        return _resp(_SEMANTIC_PASS)

    return _route


def _mech_pass() -> dict[str, Any]:
    return {
        "tier": 3,
        "tier_label": "model_assisted",
        "mechanical_check": "passed",
        "semantic_check": "skipped",
        "failure_reason": None,
    }


# Two chunks from two distinct domains — the multi-domain (Tier 2) setup.
_TWO_DOMAIN_CHUNKS = [
    {
        "chunk_id": "doc_1_chunk_A",
        "text": "The committee approved the measure by a wide margin.",
        "source_url": "https://alpha.example.com/a",
        "domain": "alpha.example.com",
    },
    {
        "chunk_id": "doc_2_chunk_B",
        "text": "The committee rejected the measure outright.",
        "source_url": "https://beta.example.org/b",
        "domain": "beta.example.org",
    },
]

_TWO_DOMAIN_DRAFT = [
    {
        "sentence_id": "s_01",
        "text": "The committee decided on the measure.",
        "is_cited": True,
        "citations": [
            {
                "citation_id": "cite_1",
                "chunk_id": "doc_1_chunk_A",
                "exact_source_quote": "The committee approved the measure by a wide margin.",
            },
            {
                "citation_id": "cite_2",
                "chunk_id": "doc_2_chunk_B",
                "exact_source_quote": "The committee rejected the measure outright.",
            },
        ],
    }
]


def _state() -> dict[str, Any]:
    state = make_initial_state(
        request_id="req",
        user_query="what did the committee decide?",
        app_config={"expertise_level": "intermediate", "banned_domains": []},
        models_config={"synthesizer": "mock/s", "verifier": "mock/v"},
        pipeline_config={"stages": {"semantic_verification_enabled": True}},
    )
    state["indexed_chunks"] = _TWO_DOMAIN_CHUNKS
    state["draft_sentences"] = _TWO_DOMAIN_DRAFT
    state["mechanical_results"] = {"cite_1": _mech_pass(), "cite_2": _mech_pass()}
    return state


async def _run_semantic(state: dict[str, Any], router: Any) -> dict[str, Any]:
    with patch(
        "axiom_rag_engine.nodes.semantic.litellm.acompletion", new_callable=AsyncMock
    ) as mock:
        mock.side_effect = router
        return await semantic_verifier_node(state)


class TestContradictionGate:
    async def test_disabled_by_default_no_tier6(self, monkeypatch) -> None:
        """Without the flag, multi-domain stays Tier 2 and no contradiction call runs."""
        get_settings.cache_clear()  # env unset -> contradiction_detection_enabled False
        with patch(
            "axiom_rag_engine.nodes.semantic.litellm.acompletion", new_callable=AsyncMock
        ) as mock:
            mock.side_effect = _router('{"contradicted": true}')  # would flag IF called
            result = await semantic_verifier_node(_state())
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 2
        # No contradiction-gate call ran (the phrase is unique to that prompt; the
        # semantic prompt separately mentions "contradiction tiers").
        systems = [c.kwargs["messages"][0]["content"].lower() for c in mock.call_args_list]
        assert not any("contradict each other" in s for s in systems)

    async def test_enabled_and_contradicted_assigns_tier6(self, monkeypatch) -> None:
        monkeypatch.setenv("AXIOM_CONTRADICTION_DETECTION_ENABLED", "true")
        get_settings.cache_clear()
        result = await _run_semantic(
            _state(), _router('{"contradicted": true, "reasoning": "approve vs reject"}')
        )
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 6
        assert vr["tier_label"] == "conflicted"
        assert any(e["event_type"] == "contradiction_result" for e in result["audit_trail"])

    async def test_enabled_and_not_contradicted_keeps_tier2(self, monkeypatch) -> None:
        monkeypatch.setenv("AXIOM_CONTRADICTION_DETECTION_ENABLED", "true")
        get_settings.cache_clear()
        result = await _run_semantic(
            _state(), _router('{"contradicted": false, "reasoning": "different aspects"}')
        )
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 2

    async def test_check_error_fails_safe_keeps_tier2(self, monkeypatch) -> None:
        """A contradiction-check error must not fabricate a Tier 6 verdict."""
        monkeypatch.setenv("AXIOM_CONTRADICTION_DETECTION_ENABLED", "true")
        get_settings.cache_clear()

        async def _route(**kwargs: Any) -> MagicMock:
            system = kwargs["messages"][0]["content"].lower()
            if "contradict each other" in system:
                raise RuntimeError("verifier down")
            return _resp(_SEMANTIC_PASS)

        result = await _run_semantic(_state(), _route)
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 2  # provisional tier kept
        assert any(e["event_type"] == "contradiction_error" for e in result["audit_trail"])

    async def test_contradiction_overrides_corroboration(self, monkeypatch) -> None:
        """With both gates on, a contradiction wins: Tier 6, and corroboration is
        never consulted (conflicting sources cannot corroborate)."""
        monkeypatch.setenv("AXIOM_CONTRADICTION_DETECTION_ENABLED", "true")
        monkeypatch.setenv("AXIOM_CORROBORATION_ENABLED", "true")
        get_settings.cache_clear()
        with patch(
            "axiom_rag_engine.nodes.semantic.litellm.acompletion", new_callable=AsyncMock
        ) as mock:
            mock.side_effect = _router(
                '{"contradicted": true}', corroboration_json='{"corroborated": false}'
            )
            result = await semantic_verifier_node(_state())
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 6
        systems = [c.kwargs["messages"][0]["content"].lower() for c in mock.call_args_list]
        assert not any("corroborate" in s for s in systems)  # short-circuited
