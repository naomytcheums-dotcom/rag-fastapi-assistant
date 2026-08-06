"""
Failure analysis -- retrieval-failure category.

For every test-set question whose gold document didn't make it into the
final top-k (see results/baseline.json), this diagnoses *where* in the
pipeline it went missing:

  - not_found_by_either_search: neither BM25 nor semantic search surfaced
    it at all -- an embedding/chunking/keyword problem.
  - lost_in_fusion_pool_size: found by one method, but ranked too low to
    survive into the reranking candidate pool -- a pool-size problem.
  - demoted_by_reranker: made it into the reranking pool, but the
    cross-encoder scored it below other candidates -- a reranker
    calibration problem, or a sign the "gold" label itself is too narrow.

Each stage points to a different fix, which is the point of separating
them instead of just reporting a single Recall@k number.

Generation-failure, citation-failure, and comprehension-failure categories
require an LLM (Claude) to judge faithfulness and relevance, and are added
by a separate pass once ANTHROPIC_API_KEY has credit -- this script only
covers what's measurable for free from the retriever itself.
"""

import json
from pathlib import Path

from retrieval import BM25_CANDIDATES, Retriever, SEMANTIC_CANDIDATES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

POOL_SIZE = max(SEMANTIC_CANDIDATES, BM25_CANDIDATES)


def _rank_of_gold(chunk_ids, chunk_by_id, gold_paths):
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        if chunk_by_id[chunk_id]["path"] in gold_paths:
            return rank
    return None


def diagnose(retriever, question, gold_paths):
    semantic_ids = retriever._semantic_search(question, SEMANTIC_CANDIDATES)
    bm25_ids = retriever._bm25_search(question, BM25_CANDIDATES)
    fused = retriever._reciprocal_rank_fusion([semantic_ids, bm25_ids])
    fused_ids = [chunk_id for chunk_id, _ in fused]

    semantic_rank = _rank_of_gold(semantic_ids, retriever.chunk_by_id, gold_paths)
    bm25_rank = _rank_of_gold(bm25_ids, retriever.chunk_by_id, gold_paths)
    fused_rank = _rank_of_gold(fused_ids, retriever.chunk_by_id, gold_paths)

    made_rerank_pool = fused_rank is not None and fused_rank <= POOL_SIZE

    if semantic_rank is None and bm25_rank is None:
        stage = "not_found_by_either_search"
        explanation = (
            "Neither semantic search nor BM25 surfaced the gold document in their "
            f"top-{SEMANTIC_CANDIDATES}/{BM25_CANDIDATES} candidates -- the chunk's "
            "embedding and its exact keywords both missed the query."
        )
    elif made_rerank_pool:
        stage = "demoted_by_reranker"
        explanation = (
            "The gold chunk reached the reranking candidate pool but the cross-encoder "
            "scored other chunks higher -- either a reranker miscalibration, or the "
            "\"wrong\" chunks are genuinely just as relevant (an annotation issue, not "
            "a retrieval bug)."
        )
    else:
        stage = "lost_in_fusion_pool_size"
        explanation = (
            f"The gold chunk was found (semantic rank={semantic_rank}, BM25 rank={bm25_rank}) "
            f"but ranked below the top-{POOL_SIZE} cutoff used to build the reranking pool."
        )

    return {
        "semantic_rank": semantic_rank,
        "bm25_rank": bm25_rank,
        "fused_rank": fused_rank,
        "stage": stage,
        "explanation": explanation,
    }


def main():
    baseline_path = RESULTS_DIR / "baseline.json"
    if not baseline_path.exists():
        raise SystemExit(f"{baseline_path} not found -- run src/evaluation.py first.")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    misses = [q for q in baseline["per_question"] if not q["hit_at_k"]]

    print(f"Diagnosing {len(misses)} retrieval misses...")
    retriever = Retriever()

    diagnoses = []
    for q in misses:
        gold_paths = set(q["gold_documents"])
        diag = diagnose(retriever, q["question"], gold_paths)
        diagnoses.append({**q, "diagnosis": diag})
        print(f"  [{q['id']}] {diag['stage']} (semantic={diag['semantic_rank']}, bm25={diag['bm25_rank']})")

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "failure_analysis_retrieval.json"
    output_path.write_text(json.dumps(diagnoses, indent=2, ensure_ascii=False), encoding="utf-8")

    stage_counts = {}
    for d in diagnoses:
        stage_counts[d["diagnosis"]["stage"]] = stage_counts.get(d["diagnosis"]["stage"], 0) + 1

    print("\nBy stage:")
    for stage, count in stage_counts.items():
        print(f"  {stage}: {count}")
    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    main()
