"""
Unit tests for the pure scoring/fusion functions in src/retrieval.py.

These are static/class methods called directly on Retriever without
instantiating it, so they run fast with no model loading, no ChromaDB, and
no BM25 index -- exactly the kind of cheap, fast test that would have
caught the RERANK_WEIGHT=0.7 regression's mechanism (score blending) in
isolation, ahead of the full 50-question evaluation catching its effect.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retrieval import Retriever, tokenize  # noqa: E402


def test_tokenize_lowercases_and_splits_on_non_alnum():
    assert tokenize("HTTPException(status_code=404)") == ["httpexception", "status_code", "404"]


def test_tokenize_empty_string():
    assert tokenize("") == []


def test_reciprocal_rank_fusion_rewards_consensus_top_rank():
    semantic = ["a", "b", "c"]
    bm25 = ["a", "c", "b"]
    fused = Retriever._reciprocal_rank_fusion([semantic, bm25])
    fused_ids = [chunk_id for chunk_id, _ in fused]
    assert fused_ids[0] == "a"  # ranked #1 by both methods


def test_reciprocal_rank_fusion_includes_ids_found_by_only_one_method():
    fused = Retriever._reciprocal_rank_fusion([["a", "b"], ["c"]])
    fused_ids = {chunk_id for chunk_id, _ in fused}
    assert fused_ids == {"a", "b", "c"}


def test_min_max_normalize_scales_to_unit_range():
    assert Retriever._min_max_normalize([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]


def test_min_max_normalize_handles_all_equal_values():
    assert Retriever._min_max_normalize([5.0, 5.0, 5.0]) == [0.5, 0.5, 0.5]


def test_min_max_normalize_handles_empty_list():
    assert Retriever._min_max_normalize([]) == []


def test_blend_scores_at_weight_one_ignores_rrf_signal():
    # candidate 0 wins on rerank score alone, candidate 1 wins on rrf alone
    blended = Retriever._blend_scores([1.0, 0.0], [0.0, 1.0], rerank_weight=1.0)
    assert blended[0] > blended[1]


def test_blend_scores_at_weight_zero_ignores_reranker_signal():
    blended = Retriever._blend_scores([1.0, 0.0], [0.0, 1.0], rerank_weight=0.0)
    assert blended[0] < blended[1]


def test_blend_scores_strong_rrf_consensus_can_outweigh_a_near_tied_rerank_gap():
    # Mirrors the real q040 case from failure_analysis.md: a candidate that
    # both search methods ranked #1 should be hard for the reranker alone
    # to fully veto once fusion consensus counts for something. Three
    # candidates on purpose: with only two, min-max normalization always
    # maps them to exactly {0, 1} regardless of how close the raw scores
    # were, which would hide the "near-tied" gap this test needs.
    rerank_scores = [0.50, 0.49, 0.0]  # A barely ahead of B; C is a clear loser
    rrf_scores = [0.0, 1.0, 0.5]  # B has full fusion consensus, A has none
    blended = Retriever._blend_scores(rerank_scores, rrf_scores, rerank_weight=0.5)
    assert blended[1] > blended[0] > blended[2]
