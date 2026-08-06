"""
Hybrid retrieval: combines dense semantic search (ChromaDB) with sparse
keyword search (BM25) via Reciprocal Rank Fusion, then reranks the fused
candidates with a cross-encoder before returning the final top-k chunks.

Hybrid search matters for a technical doc set specifically because exact
identifiers ("async def", "HTTPException", "Depends") carry meaning that a
dense embedding can blur, while BM25 finds them precisely -- but BM25 alone
misses paraphrased questions a user actually types. RRF combines both
rankings without needing to reconcile their very different score scales.
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

# Must be set before importing transformers/sentence_transformers: if a
# TensorFlow install with an incompatible Keras 3 happens to be present on
# the machine (as on this one), transformers tries to load its unused TF
# integration and crashes the whole import chain. We only need the PyTorch
# backend, so this is set unconditionally rather than relying on every
# entry point (CLI, Streamlit, tests) to remember an env var.
os.environ.setdefault("USE_TF", "0")

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "fastapi_docs"

SEMANTIC_CANDIDATES = 20
BM25_CANDIDATES = 20
RRF_K = 60  # standard smoothing constant from the original RRF paper
FINAL_TOP_K = 5

# Weight given to the cross-encoder's (normalized) score when blending it
# with the RRF fusion score to produce the final ranking. 1.0 would mean
# "trust the reranker completely and ignore fusion order", which is what
# this system did before failure analysis found it demoting documents that
# both BM25 and semantic search had ranked #1 (see
# results/failure_analysis.md, category 1) -- a generic MS MARCO reranker
# rewarding lexical density over topical correctness on this technical doc
# set. 0.7 keeps the reranker as the dominant signal while letting a
# strong fusion consensus pull a candidate back up.
RERANK_WEIGHT = 0.85

# Defensive cap: an unbounded user-typed query gets fed to the cross-encoder
# once per candidate (up to POOL_SIZE times), so a pathologically long
# query is a cheap way to inflate reranking cost per request.
MAX_QUERY_CHARS = 500

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

logger = logging.getLogger(__name__)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class Retriever:
    def __init__(self):
        chunks_path = PROCESSED_DIR / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(f"{chunks_path} not found -- run src/indexing.py first.")

        try:
            self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{chunks_path} is not valid JSON (corrupted or truncated write?) -- "
                "re-run src/indexing.py to regenerate it."
            ) from exc

        self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}

        tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_ids = [c["chunk_id"] for c in self.chunks]

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            self.collection = client.get_collection(COLLECTION_NAME)
        except Exception as exc:
            # chromadb has changed which exception type this raises across
            # versions (ValueError historically) -- catch broadly and
            # convert to a specific, actionable error rather than leaking
            # whatever internal exception this version happens to use.
            raise RuntimeError(
                f"Vector store collection '{COLLECTION_NAME}' not found at {CHROMA_DIR} -- "
                "the vector store is empty or missing. Run src/indexing.py first."
            ) from exc

        logger.info("Loaded %d chunks, BM25 index, and Chroma collection '%s'", len(self.chunks), COLLECTION_NAME)

        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.reranker = CrossEncoder(CROSS_ENCODER_MODEL_NAME)

    def _semantic_search(self, query, n):
        query_embedding = self.embedder.encode([query]).tolist()
        try:
            results = self.collection.query(query_embeddings=query_embedding, n_results=n)
        except Exception as exc:
            raise RuntimeError(f"Semantic search against the vector store failed: {exc}") from exc
        return results["ids"][0]

    def _bm25_search(self, query, n):
        scores = self.bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.bm25_ids[i] for i in ranked_indices]

    @staticmethod
    def _reciprocal_rank_fusion(ranked_id_lists, k=RRF_K):
        fused_scores = {}
        for ranked_ids in ranked_id_lists:
            for rank, chunk_id in enumerate(ranked_ids):
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

    @staticmethod
    def _min_max_normalize(values):
        if not values:
            return []
        lo, hi = min(values), max(values)
        if hi == lo:
            return [0.5 for _ in values]
        return [(v - lo) / (hi - lo) for v in values]

    @classmethod
    def _blend_scores(cls, rerank_scores, rrf_scores, rerank_weight=RERANK_WEIGHT):
        norm_rerank = cls._min_max_normalize(rerank_scores)
        norm_rrf = cls._min_max_normalize(rrf_scores)
        return [rerank_weight * r + (1 - rerank_weight) * f for r, f in zip(norm_rerank, norm_rrf)]

    def retrieve(self, query, top_k=FINAL_TOP_K):
        query = query.strip()[:MAX_QUERY_CHARS]
        if not query:
            raise ValueError("query must not be empty")

        start = time.perf_counter()
        semantic_ids = self._semantic_search(query, SEMANTIC_CANDIDATES)
        bm25_ids = self._bm25_search(query, BM25_CANDIDATES)

        fused = self._reciprocal_rank_fusion([semantic_ids, bm25_ids])
        fused_score_by_id = dict(fused)
        candidate_ids = [chunk_id for chunk_id, _ in fused[: max(SEMANTIC_CANDIDATES, BM25_CANDIDATES)]]
        candidates = [self.chunk_by_id[cid] for cid in candidate_ids]

        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = list(self.reranker.predict(pairs)) if pairs else []
        rrf_scores = [fused_score_by_id[cid] for cid in candidate_ids]
        blended_scores = self._blend_scores(rerank_scores, rrf_scores)

        reranked = sorted(
            zip(candidates, rerank_scores, blended_scores), key=lambda item: item[2], reverse=True
        )
        results = [
            {**chunk, "rerank_score": float(rerank_score), "blended_score": float(blended_score)}
            for chunk, rerank_score, blended_score in reranked[:top_k]
        ]

        elapsed_ms = round((time.perf_counter() - start) * 1000)
        logger.info(
            "retrieve() query=%r -> %d results in %dms (top path=%s)",
            query[:80], len(results), elapsed_ms, results[0]["path"] if results else None,
        )
        return results


def main():
    parser = argparse.ArgumentParser(description="Query the FastAPI docs retriever from the command line.")
    parser.add_argument("query", help="Natural-language question")
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K)
    args = parser.parse_args()

    print("Loading retriever (BM25 index, embedding model, cross-encoder)...")
    retriever = Retriever()
    results = retriever.retrieve(args.query, top_k=args.top_k)

    for i, chunk in enumerate(results, start=1):
        print(f"\n--- Result {i} (rerank_score={chunk['rerank_score']:.4f}) ---")
        print(f"path: {chunk['path']} | heading: {chunk['heading']}")
        print(chunk["text"][:300])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
