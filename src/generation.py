"""
Generation stage: builds a grounded prompt from retrieved chunks and calls
the Claude API to produce a cited answer.

The system prompt is deliberately strict about answering only from the
provided context and citing every claim -- that constraint is what the
failure-analysis stage later checks for (faithfulness / hallucination rate).
"""

import argparse
import os
from pathlib import Path

from anthropic import Anthropic

from retrieval import FINAL_TOP_K, Retriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = os.environ.get("RAG_GENERATION_MODEL", "claude-sonnet-5")
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a technical assistant that answers questions about FastAPI using ONLY the documentation excerpts provided in the user message. You have no knowledge beyond what is given in those excerpts.

Rules:
- Answer only using information present in the numbered context excerpts.
- Every factual claim in your answer must be followed by a citation like [1] or [2], referring to the excerpt number it came from.
- If the context does not contain enough information to answer, say so explicitly instead of guessing.
- Prefer quoting exact function, parameter, and class names as they appear in the excerpts.
- Keep the answer concise and technically precise."""


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
        key, value = key.strip(), value.strip().strip('"').strip("'")
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
        self.client = Anthropic(api_key=api_key)

    def generate(self, question, chunks):
        user_prompt = build_user_prompt(question, chunks)
        response = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        answer_text = "".join(block.text for block in response.content if block.type == "text")

        sources = [
            {"index": i, "path": c["path"], "heading": c["heading"], "doc_title": c["doc_title"]}
            for i, c in enumerate(chunks, start=1)
        ]

        return {"answer": answer_text, "sources": sources, "model": MODEL_NAME}


def main():
    parser = argparse.ArgumentParser(description="Ask the FastAPI RAG assistant a question end-to-end.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K)
    args = parser.parse_args()

    print("Loading retriever (BM25 index, embedding model, cross-encoder)...")
    retriever = Retriever()
    chunks = retriever.retrieve(args.question, top_k=args.top_k)

    print("Generating answer...")
    generator = Generator()
    result = generator.generate(args.question, chunks)

    print(f"\nQuestion: {args.question}\n")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  [{s['index']}] {s['path']} -- {s['heading'] or s['doc_title']}")


if __name__ == "__main__":
    main()
