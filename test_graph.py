"""
Phase 4 — TDD: LangGraph wiring, route_post_verification, and end-to-end loop tests.

Test categories:
  A. route_post_verification — unit tests for routing logic
  B. Graph structure — node/edge validation
  C. End-to-end loop tests — mocked LLM, verifying the graph catches
     hallucinations and routes backward for rewrite
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from graph import route_post_verification, build_axiom_graph
from state import make_initial_state

# Synthesizer and Semantic models used in tests — must match _base_state.
_SYNTH_MODEL = "claude-3-5-sonnet-20241022"
_VERIFIER_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_litellm_response(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_model_router(
    synth_responses: list[str],
    semantic_responses: list[str],
) -> MagicMock:
    """
    Returns a side_effect callable that routes litellm.completion calls
    based on the `model` kwarg. This avoids the module-singleton patch
    collision where nodes.synthesizer.litellm and nodes.semantic.litellm
    are the same object.
    """
    synth_q: deque[str] = deque(synth_responses)
    semantic_q: deque[str] = deque(semantic_responses)

    def _router(*args: Any, **kwargs: Any) -> MagicMock:
        model = kwargs.get("model") or (args[0] if args else "")
        if model == _SYNTH_MODEL:
            if not synth_q:
                raise RuntimeError("Unexpected extra synthesizer call")
            return _mock_litellm_response(synth_q.popleft())
        elif model == _VERIFIER_MODEL:
            if not semantic_q:
                raise RuntimeError("Unexpected extra semantic call")
            return _mock_litellm_response(semantic_q.popleft())
        else:
            raise RuntimeError(f"Unexpected model in litellm.completion: {model!r}")

    return _router


def _base_state(**overrides: Any) -> dict[str, Any]:
    state = make_initial_state(
        request_id="req_test",
        user_query="What is a solid-state battery?",
        app_config={"expertise_level": "intermediate", "banned_domains": []},
        models_config={
            "synthesizer": _SYNTH_MODEL,
            "verifier": _VERIFIER_MODEL,
        },
        pipeline_config={
            "stages": {
                "semantic_verification_enabled": True,
                "max_ranked_chunks": 10,
                "max_rewrite_loops": 3,
            }
        },
    )
    state.update(overrides)
    return state


_SAMPLE_CHUNKS = [
    {
        "chunk_id": "doc_1_chunk_A",
        "text": (
            "Solid-state batteries replace liquid electrolytes with solid ceramics. "
            "This substitution significantly improves thermal stability and energy density."
        ),
        "source_url": "https://example.com/batteries",
        "is_authoritative": True,
    },
]


# ===========================================================================
# A. route_post_verification — unit tests
# ===========================================================================


class TestRoutePostVerification:
    def test_ends_when_unanswerable(self) -> None:
        state = _base_state(is_answerable=False)
        assert route_post_verification(state) == "__end__"

    def test_ends_when_loop_exhausted(self) -> None:
        state = _base_state(loop_count=3, pending_rewrite_count=1)
        assert route_post_verification(state) == "__end__"

    def test_ends_when_loop_exceeds_max(self) -> None:
        state = _base_state(loop_count=5, pending_rewrite_count=1)
        assert route_post_verification(state) == "__end__"

    def test_loops_back_on_pending_rewrites(self) -> None:
        state = _base_state(loop_count=1, pending_rewrite_count=1)
        assert route_post_verification(state) == "synthesizer"

    def test_ends_when_all_passed(self) -> None:
        state = _base_state(loop_count=1, pending_rewrite_count=0)
        assert route_post_verification(state) == "__end__"

    def test_ends_with_default_is_answerable_true(self) -> None:
        """is_answerable defaults to True when missing from state."""
        state = _base_state(loop_count=1, pending_rewrite_count=0)
        del state["is_answerable"]
        assert route_post_verification(state) == "__end__"

    def test_respects_custom_max_rewrite_loops(self) -> None:
        state = _base_state(loop_count=2, pending_rewrite_count=1)
        state["pipeline_config"]["stages"]["max_rewrite_loops"] = 2
        assert route_post_verification(state) == "__end__"

    def test_loops_when_under_custom_max(self) -> None:
        state = _base_state(loop_count=1, pending_rewrite_count=1)
        state["pipeline_config"]["stages"]["max_rewrite_loops"] = 5
        assert route_post_verification(state) == "synthesizer"

    def test_unanswerable_takes_priority_over_pending_rewrites(self) -> None:
        """Even with pending rewrites, unanswerable should END immediately."""
        state = _base_state(
            is_answerable=False,
            loop_count=0,
            pending_rewrite_count=2,
        )
        assert route_post_verification(state) == "__end__"


# ===========================================================================
# B. Graph structure — node/edge validation
# ===========================================================================


class TestGraphStructure:
    def test_graph_compiles(self) -> None:
        graph = build_axiom_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self) -> None:
        graph = build_axiom_graph()
        node_names = set(graph.get_graph().nodes.keys())
        # LangGraph adds __start__ and __end__ nodes.
        assert "synthesizer" in node_names
        assert "verifier" in node_names


# ===========================================================================
# C. End-to-end loop tests (mocked LLM)
# ===========================================================================


class TestEndToEndLoop:
    """
    These tests invoke the compiled graph with mocked LLM calls.
    They verify that:
      - A correct citation flows through in one pass.
      - A hallucinated citation triggers a rewrite loop.
      - The escape hatch terminates early.
      - Loop exhaustion terminates after max_rewrite_loops.

    All tests use a single @patch("litellm.completion") with a model-based
    router, because nodes.synthesizer.litellm and nodes.semantic.litellm
    are the same module-singleton — dual patching causes one to overwrite
    the other.
    """

    @patch("litellm.completion")
    def test_happy_path_single_pass(self, mock_llm: MagicMock) -> None:
        """Correct verbatim quote → mechanical pass → semantic pass → END."""
        synth_json = json.dumps({
            "is_answerable": True,
            "sentences": [
                {
                    "sentence_id": "s_01",
                    "text": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
                    "is_cited": True,
                    "citations": [
                        {
                            "citation_id": "cite_1",
                            "chunk_id": "doc_1_chunk_A",
                            "exact_source_quote": (
                                "Solid-state batteries replace liquid electrolytes with solid ceramics."
                            ),
                        }
                    ],
                }
            ],
        })
        semantic_json = json.dumps({
            "tier": 1,
            "semantic_check": "passed",
            "failure_reason": None,
            "reasoning": "Claim faithfully represents the authoritative source.",
        })
        mock_llm.side_effect = _make_model_router([synth_json], [semantic_json])

        graph = build_axiom_graph()
        result = graph.invoke(_base_state(indexed_chunks=_SAMPLE_CHUNKS))

        assert result["is_answerable"] is True
        assert len(result["final_sentences"]) == 1
        assert result["final_sentences"][0]["verification"]["tier"] == 1
        assert result["loop_count"] == 1

    @patch("litellm.completion")
    def test_hallucination_triggers_rewrite_loop(self, mock_llm: MagicMock) -> None:
        """
        Pass 1: Fabricated quote → Mechanical Tier 5 → loops back.
        Pass 2: Corrected verbatim quote → passes → END.
        """
        hallucinated_json = json.dumps({
            "is_answerable": True,
            "sentences": [
                {
                    "sentence_id": "s_01",
                    "text": "Solid-state batteries use ceramic electrolytes.",
                    "is_cited": True,
                    "citations": [
                        {
                            "citation_id": "cite_1",
                            "chunk_id": "doc_1_chunk_A",
                            "exact_source_quote": "This quote is completely fabricated and does not exist.",
                        }
                    ],
                }
            ],
        })
        corrected_json = json.dumps({
            "is_answerable": True,
            "sentences": [
                {
                    "sentence_id": "s_01",
                    "text": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
                    "is_cited": True,
                    "citations": [
                        {
                            "citation_id": "cite_1",
                            "chunk_id": "doc_1_chunk_A",
                            "exact_source_quote": (
                                "Solid-state batteries replace liquid electrolytes with solid ceramics."
                            ),
                        }
                    ],
                }
            ],
        })
        semantic_json = json.dumps({
            "tier": 1,
            "semantic_check": "passed",
            "failure_reason": None,
            "reasoning": "Claim faithfully represents the source.",
        })
        mock_llm.side_effect = _make_model_router(
            [hallucinated_json, corrected_json],
            [semantic_json],  # Only called on pass 2 (pass 1 fails mechanical)
        )

        graph = build_axiom_graph()
        result = graph.invoke(_base_state(indexed_chunks=_SAMPLE_CHUNKS))

        assert result["is_answerable"] is True
        assert len(result["final_sentences"]) == 1
        assert result["final_sentences"][0]["verification"]["tier"] == 1
        assert result["loop_count"] == 2

    @patch("litellm.completion")
    def test_escape_hatch_terminates_immediately(self, mock_llm: MagicMock) -> None:
        """is_answerable=false → END without entering verification."""
        unanswerable_json = json.dumps({
            "is_answerable": False,
            "sentences": [],
        })
        mock_llm.side_effect = _make_model_router([unanswerable_json], [])

        graph = build_axiom_graph()
        result = graph.invoke(_base_state(indexed_chunks=_SAMPLE_CHUNKS))

        assert result["is_answerable"] is False
        assert result.get("final_sentences", []) == [] or result["final_sentences"] == []

    @patch("litellm.completion")
    def test_loop_exhaustion_terminates_after_max_loops(self, mock_llm: MagicMock) -> None:
        """
        Synthesizer keeps hallucinating. After 3 loops the graph terminates.
        """
        hallucinated_json = json.dumps({
            "is_answerable": True,
            "sentences": [
                {
                    "sentence_id": "s_01",
                    "text": "Some claim.",
                    "is_cited": True,
                    "citations": [
                        {
                            "citation_id": "cite_1",
                            "chunk_id": "doc_1_chunk_A",
                            "exact_source_quote": "This fabricated quote never appears in the chunk.",
                        }
                    ],
                }
            ],
        })
        # 3 synthesis calls, all hallucinated. No semantic calls (mechanical fails).
        mock_llm.side_effect = _make_model_router(
            [hallucinated_json] * 3,
            [],
        )

        graph = build_axiom_graph()
        result = graph.invoke(_base_state(indexed_chunks=_SAMPLE_CHUNKS))

        assert result["loop_count"] == 3

    @patch("litellm.completion")
    def test_tier_4_semantic_failure_triggers_rewrite(self, mock_llm: MagicMock) -> None:
        """
        Mechanical passes but semantic finds misrepresentation (Tier 4).
        Graph loops back for a rewrite, second pass succeeds.
        """
        synth_json = json.dumps({
            "is_answerable": True,
            "sentences": [
                {
                    "sentence_id": "s_01",
                    "text": "Solid-state batteries have perfect thermal stability.",
                    "is_cited": True,
                    "citations": [
                        {
                            "citation_id": "cite_1",
                            "chunk_id": "doc_1_chunk_A",
                            "exact_source_quote": (
                                "This substitution significantly improves thermal stability and energy density."
                            ),
                        }
                    ],
                }
            ],
        })
        corrected_json = json.dumps({
            "is_answerable": True,
            "sentences": [
                {
                    "sentence_id": "s_01",
                    "text": "This substitution significantly improves thermal stability and energy density.",
                    "is_cited": True,
                    "citations": [
                        {
                            "citation_id": "cite_1",
                            "chunk_id": "doc_1_chunk_A",
                            "exact_source_quote": (
                                "This substitution significantly improves thermal stability and energy density."
                            ),
                        }
                    ],
                }
            ],
        })
        tier4_json = json.dumps({
            "tier": 4,
            "semantic_check": "failed",
            "failure_reason": "Claim overstates 'improves' as 'perfect'.",
            "reasoning": "The source says 'improves' not 'perfect'.",
        })
        tier1_json = json.dumps({
            "tier": 1,
            "semantic_check": "passed",
            "failure_reason": None,
            "reasoning": "Claim now faithfully represents the source.",
        })
        mock_llm.side_effect = _make_model_router(
            [synth_json, corrected_json],
            [tier4_json, tier1_json],
        )

        graph = build_axiom_graph()
        result = graph.invoke(_base_state(indexed_chunks=_SAMPLE_CHUNKS))

        assert result["is_answerable"] is True
        assert result["final_sentences"][0]["verification"]["tier"] == 1

    @patch("litellm.completion")
    def test_audit_trail_accumulates_across_loops(self, mock_llm: MagicMock) -> None:
        """The audit_trail must accumulate events from every pass, never overwrite."""
        hallucinated_json = json.dumps({
            "is_answerable": True,
            "sentences": [
                {
                    "sentence_id": "s_01",
                    "text": "Claim.",
                    "is_cited": True,
                    "citations": [
                        {
                            "citation_id": "cite_1",
                            "chunk_id": "doc_1_chunk_A",
                            "exact_source_quote": "Fabricated nonsense that does not exist in chunk.",
                        }
                    ],
                }
            ],
        })
        corrected_json = json.dumps({
            "is_answerable": True,
            "sentences": [
                {
                    "sentence_id": "s_01",
                    "text": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
                    "is_cited": True,
                    "citations": [
                        {
                            "citation_id": "cite_1",
                            "chunk_id": "doc_1_chunk_A",
                            "exact_source_quote": (
                                "Solid-state batteries replace liquid electrolytes with solid ceramics."
                            ),
                        }
                    ],
                }
            ],
        })
        semantic_json = json.dumps({
            "tier": 1, "semantic_check": "passed", "failure_reason": None, "reasoning": "ok"
        })
        mock_llm.side_effect = _make_model_router(
            [hallucinated_json, corrected_json],
            [semantic_json],
        )

        graph = build_axiom_graph()
        result = graph.invoke(_base_state(indexed_chunks=_SAMPLE_CHUNKS))

        trail = result.get("audit_trail", [])
        assert len(trail) > 0
        nodes_in_trail = {e["node"] for e in trail}
        assert "synthesizer" in nodes_in_trail
        assert "verifier" in nodes_in_trail
