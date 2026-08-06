"""
Regression gate: fails (exit code 1) if current retrieval quality has
dropped more than REGRESSION_THRESHOLD relative to the last accepted
baseline in results/regression_baseline.json.

Meant to run in CI on every push/PR (see
.github/workflows/regression.yml) so a retrieval-affecting change can't
silently make the system worse. This is exactly the check that would have
caught the RERANK_WEIGHT=0.7 regression during failure analysis before it
was accepted (see results/failure_analysis.md) -- that change was only
rejected because it happened to be measured against the full test set by
hand; this makes that check automatic and mandatory.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation import evaluate_retrieval, summarize  # noqa: E402
from retrieval import Retriever  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
BASELINE_PATH = RESULTS_DIR / "regression_baseline.json"

REGRESSION_THRESHOLD = 0.05  # 5% relative drop blocks the change


def load_baseline_summary():
    if not BASELINE_PATH.exists():
        return None
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["summary"]


def check_metric(name, current, baseline):
    if baseline is None:
        return True, f"{name}: {current:.3f} (no baseline yet -- nothing to compare against)"

    relative_change = (current - baseline) / baseline if baseline else 0.0
    ok = relative_change >= -REGRESSION_THRESHOLD
    status = "OK" if ok else "REGRESSION"
    return ok, f"{name}: {baseline:.3f} -> {current:.3f} ({relative_change:+.1%}) [{status}]"


def main():
    test_set = json.loads((DATA_DIR / "test_set.json").read_text(encoding="utf-8"))

    print("Loading retriever...")
    retriever = Retriever()

    print(f"Evaluating {len(test_set)} questions...")
    per_question = evaluate_retrieval(retriever, test_set)
    current = summarize(per_question)

    baseline = load_baseline_summary()

    passed = True
    for label, key in [("Recall@5", "recall_at_k"), ("MRR", "mrr")]:
        ok, message = check_metric(label, current[key], baseline[key] if baseline else None)
        print(message)
        passed = passed and ok

    if not passed:
        print(
            "\nREGRESSION DETECTED -- failing. If this drop is expected and "
            "accepted, update the baseline deliberately with:\n"
            "  python tests/test_regression.py --update-baseline"
        )
        sys.exit(1)

    print("\nNo regression detected.")

    if baseline is None or "--update-baseline" in sys.argv:
        RESULTS_DIR.mkdir(exist_ok=True)
        BASELINE_PATH.write_text(
            json.dumps({"summary": current, "per_question": per_question}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Baseline written -> {BASELINE_PATH}")


if __name__ == "__main__":
    main()
