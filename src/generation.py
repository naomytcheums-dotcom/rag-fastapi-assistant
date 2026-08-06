"""
Generation stage: builds a grounded prompt from retrieved chunks and calls
the Claude API to produce a cited answer.

The system prompt is deliberately strict about answering only from the
provided context and citing every claim -- that constraint is what the
failure-analysis stage later checks for (faithfulness / hallucination rate).
"""

import argparse
import logging
import os
import time
from pathlib import Path

import anthropic
from anthropic import Anthropic

from retrieval import FINAL_TOP_K, Retriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = os.environ.get("RAG_GENERATION_MODEL", "claude-sonnet-5")
MAX_TOKENS = 1024

# The SDK default timeout is measured in minutes, far too long for an
# interactive chat request -- a hung call would leave a user's dashboard
# spinner stuck with no feedback. 30s is generous for a single non-streamed
# call at this MAX_TOKENS size.
CLIENT_TIMEOUT_SECONDS = 30.0

# Approximate list pricing for cost-per-request estimates -- verify against
# https://www.anthropic.com/pricing before trusting this for real budgeting;
# it is not fetched live and will go stale.
PRICE_PER_MILLION_INPUT_TOKENS_USD = float(os.environ.get("RAG_PRICE_INPUT_PER_M", "3.0"))
PRICE_PER_MILLION_OUTPUT_TOKENS_USD = float(os.environ.get("RAG_PRICE_OUTPUT_PER_M", "15.0"))

logger = logging.getLogger(__name__)


def estimate_cost_usd(input_tokens, output_tokens):
    return (
        input_tokens * PRICE_PER_MILLION_INPUT_TOKENS_USD
        + output_tokens * PRICE_PER_MILLION_OUTPUT_TOKENS_USD
    ) / 1_000_000

SYSTEM_PROMPT = """You are a technical assistant that answers questions about FastAPI using ONLY the documentation excerpts provided in the user message. You have no knowledge beyond what is given in those excerpts.

Rules:
- Answer only using information present in the numbered context excerpts.
- Every factual claim in your answer must be followed by a citation like [1] or [2], referring to the excerpt number it came from.
- If the context does not contain enough information to answer, say so explicitly instead of guessing.
- Prefer quoting exact function, parameter, and class names as they appear in the excerpts.
- Keep the answer concise and technically precise."""


def _strip_inline_comment(value):
    """Handle `KEY=value  # comment` -- a bare `#` truncates the value,
    but a `#` inside a quoted value is kept (e.g. KEY="value#withhash")."""
    if value[:1] in ('"', "'"):
        quote = value[0]
        end = value.find(quote, 1)
        return value[1:end] if end != -1 else value.strip(quote)

    hash_index = value.find("#")
    return (value[:hash_index] if hash_index != -1 else value).strip()


def load_dotenv_if_present():
    """Minimal, dependency-free .env loader: sets KEY=VALUE lines from a
    project-root .env file into os.environ, without overriding variables
    the shell/environment already set."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), _strip_inline_comment(value.strip())
        os.environ.setdefault(key, value)


def build_context_block(chunks):
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        label = chunk["heading"] or chunk["doc_title"]
        parts.append(f"[{i}] Source: {chunk['path']} ({label})\n{chunk['text']}")
    return "\n\n".join(parts)


def build_user_prompt(question, chunks):
    context_block = build_context_block(chunks)
    return (
        f"Context excerpts from the FastAPI documentation:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above, with citations."
    )


class Generator:
    def __init__(self):
        load_dotenv_if_present()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Get a key from https://console.anthropic.com/ "
                "and either export it in your shell or put it in a .env file at the project "
                "root as ANTHROPIC_API_KEY=sk-ant-..."
            )
        self.client = Anthropic(api_key=api_key, timeout=CLIENT_TIMEOUT_SECONDS)

    def generate(self, question, chunks):
        user_prompt = build_user_prompt(question, chunks)

        start = time.perf_counter()
        try:
            response = self.client.messages.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.APITimeoutError as exc:
            raise RuntimeError(f"Claude API call timed out after {CLIENT_TIMEOUT_SECONDS}s") from exc
        except anthropic.RateLimitError as exc:
            raise RuntimeError("Claude API rate limit hit -- retry after a short backoff") from exc
        except anthropic.APIStatusError as exc:
            raise RuntimeError(f"Claude API returned an error (status {exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise RuntimeError("Could not reach the Claude API -- check network connectivity") from exc

        elapsed_ms = round((time.perf_counter() - start) * 1000)
        answer_text = "".join(block.text for block in response.content if block.type == "text")

        usage = response.usage
        logger.info(
            "generate() model=%s in_tokens=%d out_tokens=%d elapsed=%dms",
            MODEL_NAME, usage.input_tokens, usage.output_tokens, elapsed_ms,
        )

        sources = [
            {"index": i, "path": c["path"], "heading": c["heading"], "doc_title": c["doc_title"]}
            for i, c in enumerate(chunks, start=1)
        ]

        return {
            "answer": answer_text,
            "sources": sources,
            "model": MODEL_NAME,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "elapsed_ms": elapsed_ms,
        }


def main():
    parser = argparse.ArgumentParser(description="Ask the FastAPI RAG assistant a question end-to-end.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K)
    args = parser.parse_args()

    print("Loading retriever (BM25 index, embedding model, cross-encoder)...")
    try:
        retriever = Retriever()
        chunks = retriever.retrieve(args.question, top_k=args.top_k)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Retrieval failed: {exc}")

    print("Generating answer...")
    try:
        generator = Generator()
        result = generator.generate(args.question, chunks)
    except (RuntimeError, EnvironmentError) as exc:
        raise SystemExit(f"Generation failed: {exc}")

    print(f"\nQuestion: {args.question}\n")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  [{s['index']}] {s['path']} -- {s['heading'] or s['doc_title']}")

    cost = estimate_cost_usd(result["input_tokens"], result["output_tokens"])
    print(
        f"\n{result['input_tokens']} input + {result['output_tokens']} output tokens, "
        f"{result['elapsed_ms']}ms, ~${cost:.4f} (approximate list pricing)"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
