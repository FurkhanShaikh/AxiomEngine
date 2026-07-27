"""Production second-stage reranker in the ranker node.

Reranking is opt-in (AXIOM_RERANKER_MODEL). These tests mock the grading LLM so
they need no model, and check the behaviors that matter: parse tolerance, the
stable-refinement reorder, off-by-default, applied-when-configured, and fail-open
when grading errors. The quality question (does reranking rank better) is
answered by the eval on real data (BENCHMARKS.md), not here.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from axiom_rag_engine.config.settings import get_settings
from axiom_rag_engine.nodes.ranker import (
    _apply_reranker,
    _parse_rerank_grade,
    ranker_node,
)
from axiom_rag_engine.state import make_initial_state


class TestParseRerankGrade:
    def test_plain_digit(self) -> None:
        assert _parse_rerank_grade("2") == 2

    def test_prose_around_digit(self) -> None:
        assert _parse_rerank_grade("Score: 3 — directly answers") == 3

    def test_think_block_stripped(self) -> None:
        assert _parse_rerank_grade("<think>hmm 2 vs 3</think>\n1") == 1

    def test_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="no 0-3 grade"):
            _parse_rerank_grade("7")

    def test_multi_digit_not_matched(self) -> None:
        with pytest.raises(ValueError, match="no 0-3 grade"):
            _parse_rerank_grade("10")

    def test_garbage_rejected(self) -> None:
        with pytest.raises(ValueError, match="no 0-3 grade"):
            _parse_rerank_grade("relevant passage")


# ---------------------------------------------------------------------------
# _apply_reranker — the reorder logic with grading injected via patch
# ---------------------------------------------------------------------------


def _ranked(*chunk_ids: str) -> list[dict[str, Any]]:
    return [{"chunk_id": c, "text": f"text {c}", "ranking_score": 0.5} for c in chunk_ids]


async def _run_apply(ranked: list[dict], grade_map: dict[str, int], *, top_k: int = 20):
    """Patch _grade_chunk to grade by chunk_id via the passed map."""

    async def _fake_grade(query: str, text: str, model: str) -> int:
        cid = text.replace("text ", "")
        return grade_map[cid]

    audit: list[dict[str, Any]] = []
    with patch("axiom_rag_engine.nodes.ranker._grade_chunk", side_effect=_fake_grade):
        applied = await _apply_reranker("q", ranked, "fake/model", top_k, audit)
    return applied, audit


class TestApplyReranker:
    async def test_reorders_by_grade_desc(self) -> None:
        ranked = _ranked("a", "b", "c", "d")
        applied, _ = await _run_apply(ranked, {"a": 0, "b": 1, "c": 3, "d": 2})
        assert applied is True
        assert [c["chunk_id"] for c in ranked] == ["c", "d", "b", "a"]
        assert [c["rerank_grade"] for c in ranked] == [3, 2, 1, 0]

    async def test_ties_preserve_base_order(self) -> None:
        ranked = _ranked("a", "b", "c")
        await _run_apply(ranked, {"a": 2, "b": 2, "c": 2})
        assert [c["chunk_id"] for c in ranked] == ["a", "b", "c"]

    async def test_tail_beyond_top_k_untouched(self) -> None:
        ranked = _ranked("a", "b", "c", "d", "e")
        # Only rerank top 2; c,d,e keep their order regardless of grade.
        await _run_apply(ranked, {"a": 0, "b": 3}, top_k=2)
        assert [c["chunk_id"] for c in ranked] == ["b", "a", "c", "d", "e"]
        assert "rerank_grade" not in ranked[2]  # tail chunk never graded

    async def test_ranking_score_untouched(self) -> None:
        ranked = _ranked("a", "b")
        await _run_apply(ranked, {"a": 0, "b": 3})
        assert all(c["ranking_score"] == 0.5 for c in ranked)

    async def test_single_chunk_skips(self) -> None:
        ranked = _ranked("a")
        applied, audit = await _run_apply(ranked, {"a": 3})
        assert applied is False
        assert not audit

    async def test_partial_failure_sinks_that_chunk(self) -> None:
        ranked = _ranked("a", "b", "c")

        async def _grade(query: str, text: str, model: str) -> int:
            if text == "text b":
                raise RuntimeError("timeout on b")
            return 3 if text == "text c" else 1

        audit: list[dict[str, Any]] = []
        with patch("axiom_rag_engine.nodes.ranker._grade_chunk", side_effect=_grade):
            applied = await _apply_reranker("q", ranked, "fake/model", 20, audit)
        assert applied is True
        # c graded 3, a graded 1, b failed -> grade 0 (sinks last).
        assert [c["chunk_id"] for c in ranked] == ["c", "a", "b"]
        event = next(e for e in audit if e["event_type"] == "ranker_reranked")
        assert event["payload"]["grade_failures"] == 1

    async def test_total_failure_fails_open(self) -> None:
        ranked = _ranked("a", "b", "c")

        async def _boom(query: str, text: str, model: str) -> int:
            raise RuntimeError("model down")

        audit: list[dict[str, Any]] = []
        with patch("axiom_rag_engine.nodes.ranker._grade_chunk", side_effect=_boom):
            applied = await _apply_reranker("q", ranked, "fake/model", 20, audit)
        assert applied is False
        assert [c["chunk_id"] for c in ranked] == ["a", "b", "c"]  # order preserved
        assert any(e["event_type"] == "ranker_rerank_error" for e in audit)
        assert all("rerank_grade" not in c for c in ranked)


# ---------------------------------------------------------------------------
# ranker_node integration — opt-in via settings, grading LLM mocked
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, text: str, quality: float = 0.5) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "source_url": f"https://example.com/{chunk_id}",
        "domain": "example.com",
        "title": "T",
        "doc_index": int(chunk_id.split("_")[1]),
        "chunk_index": 0,
        "quality_score": quality,
    }


def _state(chunks: list[dict], query: str = "ocean tides") -> dict[str, Any]:
    s = make_initial_state(
        request_id="req", user_query=query, app_config={}, models_config={}, pipeline_config={}
    )
    s["scored_chunks"] = chunks
    return s


def _grade_response(grade: str) -> Any:
    from unittest.mock import MagicMock

    msg = MagicMock()
    msg.content = grade
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None
    return resp


class TestRerankerOptIn:
    async def test_off_by_default(self) -> None:
        get_settings.cache_clear()
        chunks = [
            _chunk("doc_1_chunk_A", "the moon causes ocean tides"),
            _chunk("doc_2_chunk_A", "gardening tips for tomatoes"),
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            result = await ranker_node(_state(chunks))
        mock_llm.assert_not_called()
        ranked = result["ranked_chunks"]
        assert all("rerank_grade" not in c for c in ranked)
        complete = next(e for e in result["audit_trail"] if e["event_type"] == "ranker_complete")
        assert complete["payload"]["ranking_mode"] == "bm25"

    async def test_applied_when_configured(self, monkeypatch) -> None:
        monkeypatch.setenv("AXIOM_RERANKER_MODEL", "fake/model")
        get_settings.cache_clear()
        # BM25 favors chunk A (query terms); grade chunk B higher so rerank flips.
        chunks = [
            _chunk("doc_1_chunk_A", "ocean tides ocean tides ocean tides"),
            _chunk("doc_2_chunk_B", "lunar gravitational pull on seawater"),
        ]

        async def _route(**kwargs: Any) -> Any:
            passage = kwargs["messages"][1]["content"]
            return _grade_response("3" if "lunar" in passage else "0")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = _route
            result = await ranker_node(_state(chunks))

        ranked = result["ranked_chunks"]
        assert ranked[0]["chunk_id"] == "doc_2_chunk_B"  # graded 3, pulled to top
        assert all("rerank_grade" in c for c in ranked)
        complete = next(e for e in result["audit_trail"] if e["event_type"] == "ranker_complete")
        assert complete["payload"]["ranking_mode"] == "bm25+rerank"

    async def test_ranking_score_untouched_by_rerank(self, monkeypatch) -> None:
        get_settings.cache_clear()
        chunks = [
            _chunk("doc_1_chunk_A", "the moon causes ocean tides on earth"),
            _chunk("doc_2_chunk_B", "the sun also contributes to tidal forces"),
        ]
        bm25 = await ranker_node(_state([dict(c) for c in chunks]))
        bm25_scores = {c["chunk_id"]: c["ranking_score"] for c in bm25["ranked_chunks"]}

        monkeypatch.setenv("AXIOM_RERANKER_MODEL", "fake/model")
        get_settings.cache_clear()
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = lambda **k: _grade_response("2")
            reranked = await ranker_node(_state([dict(c) for c in chunks]))
        rerank_scores = {c["chunk_id"]: c["ranking_score"] for c in reranked["ranked_chunks"]}
        assert bm25_scores == rerank_scores

    async def test_grading_failure_falls_back(self, monkeypatch) -> None:
        monkeypatch.setenv("AXIOM_RERANKER_MODEL", "fake/model")
        get_settings.cache_clear()
        chunks = [
            _chunk("doc_1_chunk_A", "ocean tides caused by the moon"),
            _chunk("doc_2_chunk_B", "unrelated content about cooking"),
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("model unreachable")
            result = await ranker_node(_state(chunks))
        ranked = result["ranked_chunks"]
        assert len(ranked) == 2  # ranking still produced
        assert all("rerank_grade" not in c for c in ranked)
        complete = next(e for e in result["audit_trail"] if e["event_type"] == "ranker_complete")
        assert complete["payload"]["ranking_mode"] == "bm25"  # fell back
