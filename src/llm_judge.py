"""
LLM-as-judge: scores a generated answer for faithfulness (does every claim
trace back to the retrieved context, with nothing invented?) and answer
relevance (does it actually address the question?), using Claude itself as
the judge.

NOT YET VERIFIED AGAINST A LIVE API CALL. Written and reasoned through
carefully (same request/response shape already confirmed working end-to-end
in generation.py, up to the point of a credit-balance error), but the JSON
parsing of the judge's free-form response in particular is the part most
likely to need a small fix once this actually runs -- see README for
status.

Calibration: `calibration/human_annotations.json` should hold ~20
hand-labeled (question, answer, context, human_faithfulness,
human_relevance) examples. `calibrate_judge()` runs this same judge prompt
against those examples and reports agreement with the human scores -- the
project targets >90% correlation per the original brief. That file is
currently an empty placeholder: producing it needs real generated answers
(API credit) plus actual manual review, neither of which can be faked.
"""

import json
import os
from pathlib import Path

from anthropic import Anthropic

from generation import MODEL_NAME, load_dotenv_if_present

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "calibration"

JUDGE_MAX_TOKENS = 200
CLIENT_TIMEOUT_SECONDS = 30.0

JUDGE_SYSTEM_PROMPT = """You are a strict grader evaluating a RAG system's answer against the context it was given. Score two things on a 1-5 integer scale:

- faithfulness: does every claim in the answer trace back to the provided context, with no invented facts?
- relevance: does the answer actually address the question asked, without padding or going off-topic?

Respond with ONLY a JSON object, no other text: {"faithfulness": <1-5>, "relevance": <1-5>, "reasoning": "<one sentence>"}"""


def _client():
    load_dotenv_if_present()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set -- see generation.py")
    return Anthropic(api_key=api_key, timeout=CLIENT_TIMEOUT_SECONDS)


def judge_answer(client, question, context_block, answer):
    user_prompt = (
        f"Question: {question}\n\nContext given to the system:\n{context_block}\n\n"
        f"System's answer:\n{answer}\n\nScore it."
    )
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=JUDGE_MAX_TOKENS,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"faithfulness": None, "relevance": None, "reasoning": f"unparseable judge output: {text[:200]}"}


def calibrate_judge():
    annotations_path = CALIBRATION_DIR / "human_annotations.json"
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    if not annotations:
        raise SystemExit(
            f"{annotations_path} is empty. Fill it with ~20 hand-labeled examples "
            "(see calibration/README.md) before running calibration -- these have "
            "to come from real system runs and real human review, not generated code."
        )

    client = _client()
    rows = []
    for example in annotations:
        judged = judge_answer(client, example["question"], example["context"], example["answer"])
        rows.append({
            "id": example.get("id"),
            "human_faithfulness": example["human_faithfulness"],
            "judge_faithfulness": judged.get("faithfulness"),
            "human_relevance": example["human_relevance"],
            "judge_relevance": judged.get("relevance"),
            "judge_reasoning": judged.get("reasoning"),
        })

    return rows


def summarize_agreement(rows):
    def agreement_rate(human_key, judge_key):
        scored = [r for r in rows if r[judge_key] is not None]
        if not scored:
            return None
        exact_or_close = sum(1 for r in scored if abs(r[human_key] - r[judge_key]) <= 1)
        return exact_or_close / len(scored)

    return {
        "n_examples": len(rows),
        "faithfulness_agreement": agreement_rate("human_faithfulness", "judge_faithfulness"),
        "relevance_agreement": agreement_rate("human_relevance", "judge_relevance"),
    }


if __name__ == "__main__":
    results = calibrate_judge()
    summary = summarize_agreement(results)
    print(json.dumps({"summary": summary, "rows": results}, indent=2, ensure_ascii=False))
