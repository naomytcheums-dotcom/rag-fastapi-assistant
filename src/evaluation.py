"""
Evaluation: measures retrieval quality (Recall@k, MRR) against the labeled
test set in data/test_set.json.

This runs entirely offline and free -- Recall@k and MRR only need the
retriever, no LLM calls. Generation-quality metrics that require an LLM
judge (faithfulness, answer relevance) are a separate, explicitly-opted-in
pass since they cost API credits -- see failure_analysis.py.

Tuning/held-out split: 15 of the 50 questions are marked "held_out": true
in test_set.json (stratified across difficulty). The RERANK_WEIGHT
hyperparameter in retrieval.py was originally chosen by testing against
all 50 questions -- a real data-leakage issue documented in AUDIT.md and
results/failure_analysis.md that can't be undone after the fact. This
split exists to stop it from happening again: any future parameter tuning
should use --exclude-held-out, and the held-out-only number is the more
trustworthy generalization estimate to report going forward.
"""

import argparse
import json
from pathlib import Path

from retrieval import Retriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

TOP_K = 5


def evaluate_retrieval(retriever, test_set, top_k=TOP_K):
    per_question = []

    for q in test_set:
        gold_paths = set(q["documents"])
        retrieved = retriever.retrieve(q["question"], top_k=top_k)
        retrieved_paths = [c["path"] for c in retrieved]

        hit = any(path in gold_paths for path in retrieved_paths)

        reciprocal_rank = 0.0
        for rank, path in enumerate(retrieved_paths, start=1):
            if path in gold_paths:
                reciprocal_rank = 1.0 / rank
                break

        per_question.append({
            "id": q["id"],
            "difficulty": q["difficulty"],
            "topic": q["topic"],
            "question": q["question"],
            "gold_documents": sorted(gold_paths),
            "retrieved_documents": retrieved_paths,
            "hit_at_k": hit,
            "reciprocal_rank": reciprocal_rank,
        })

    return per_question


def _average(key, subset):
    if not subset:
        return None
    return sum(item[key] for item in subset) / len(subset)


def summarize(per_question):
    summary = {
        "n_questions": len(per_question),
        "recall_at_k": _average("hit_at_k", per_question),
        "mrr": _average("reciprocal_rank", per_question),
        "by_difficulty": {},
    }

    for difficulty in ("easy", "medium", "hard"):
        subset = [q for q in per_question if q["difficulty"] == difficulty]
        summary["by_difficulty"][difficulty] = {
            "n_questions": len(subset),
            "recall_at_k": _average("hit_at_k", subset),
            "mrr": _average("reciprocal_rank", subset),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against data/test_set.json")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--output", default="baseline.json", help="Filename under results/ to write")
    parser.add_argument(
        "--held-out-only", action="store_true",
        help="Evaluate only the 15 questions marked held_out=true (see module docstring)",
    )
    parser.add_argument(
        "--exclude-held-out", action="store_true",
        help="Evaluate only the 35 tuning questions -- use this when tuning a parameter",
    )
    args = parser.parse_args()

    test_set = json.loads((DATA_DIR / "test_set.json").read_text(encoding="utf-8"))
    if args.held_out_only:
        test_set = [q for q in test_set if q["held_out"]]
    elif args.exclude_held_out:
        test_set = [q for q in test_set if not q["held_out"]]

    print("Loading retriever (BM25 index, embedding model, cross-encoder)...")
    retriever = Retriever()

    print(f"Evaluating {len(test_set)} questions (top_k={args.top_k})...")
    per_question = evaluate_retrieval(retriever, test_set, top_k=args.top_k)
    summary = summarize(per_question)

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / args.output
    output_path.write_text(
        json.dumps({"summary": summary, "per_question": per_question}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nRecall@{args.top_k}: {summary['recall_at_k']:.1%}")
    print(f"MRR: {summary['mrr']:.3f}")
    for difficulty, stats in summary["by_difficulty"].items():
        print(
            f"  {difficulty}: recall@{args.top_k}={stats['recall_at_k']:.1%}, "
            f"mrr={stats['mrr']:.3f} (n={stats['n_questions']})"
        )
    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    main()
