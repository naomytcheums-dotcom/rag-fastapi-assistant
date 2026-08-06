"""
Hallucination detection: extracts the factual claims from a generated
answer and checks each one against the retrieved context using Claude as a
verifier. Reports a hallucination rate (fraction of claims not supported
by the context); the system can be configured to withhold or flag an
answer above HALLUCINATION_THRESHOLD instead of presenting an unverified
claim as fact.

NOT YET VERIFIED AGAINST A LIVE API CALL -- same status as llm_judge.py,
see that file's docstring and AUDIT.md. This calls the API once per claim
plus once for extraction, so cost scales with answer length; that's an
inherent tradeoff of claim-level verification, not an oversight.
"""

import json
import os
from pathlib import Path

from anthropic import Anthropic

from generation import MODEL_NAME, load_dotenv_if_present

HALLUCINATION_THRESHOLD = 0.2  # flag if more than 20% of claims are unsupported
CLIENT_TIMEOUT_SECONDS = 30.0

CLAIM_EXTRACTION_SYSTEM_PROMPT = """Extract every distinct factual claim from the given answer as a JSON array of short strings. A claim is a specific, checkable statement (e.g. "HTTPException takes a status_code parameter"), not filler, opinion, or a citation marker. Respond with ONLY a JSON array of strings, no other text."""

CLAIM_VERIFICATION_SYSTEM_PROMPT = """You are given a context passage and a single claim. Respond with ONLY the single word "supported" if the claim is directly backed by the context, or "unsupported" if it is not stated or is contradicted by the context."""


def _client():
    load_dotenv_if_present()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set -- see generation.py")
    return Anthropic(api_key=api_key, timeout=CLIENT_TIMEOUT_SECONDS)


def extract_claims(client, answer):
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=500,
        system=CLAIM_EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": answer}],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    try:
        claims = json.loads(text)
        return claims if isinstance(claims, list) else []
    except json.JSONDecodeError:
        return []


def verify_claim(client, claim, context_block):
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=10,
        system=CLAIM_VERIFICATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context_block}\n\nClaim: {claim}"}],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip().lower()
    return text.startswith("supported")


def detect_hallucinations(answer, context_block):
    client = _client()
    claims = extract_claims(client, answer)
    if not claims:
        return {"claims": [], "hallucination_rate": 0.0, "flagged": False}

    results = [{"claim": claim, "supported": verify_claim(client, claim, context_block)} for claim in claims]
    unsupported_count = sum(1 for r in results if not r["supported"])
    rate = unsupported_count / len(results)

    return {
        "claims": results,
        "hallucination_rate": rate,
        "flagged": rate > HALLUCINATION_THRESHOLD,
    }


def main():
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generation import Generator, build_context_block
    from retrieval import FINAL_TOP_K, Retriever

    parser = argparse.ArgumentParser(description="Check a generated answer for hallucinated claims.")
    parser.add_argument("question")
    args = parser.parse_args()

    retriever = Retriever()
    chunks = retriever.retrieve(args.question, top_k=FINAL_TOP_K)
    context_block = build_context_block(chunks)

    generator = Generator()
    result = generator.generate(args.question, chunks)

    report = detect_hallucinations(result["answer"], context_block)

    print(f"Answer: {result['answer']}\n")
    print(f"Hallucination rate: {report['hallucination_rate']:.1%} (flagged={report['flagged']})")
    for claim in report["claims"]:
        status = "OK" if claim["supported"] else "UNSUPPORTED"
        print(f"  [{status}] {claim['claim']}")


if __name__ == "__main__":
    main()
