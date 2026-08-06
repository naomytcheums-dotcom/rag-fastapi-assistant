"""
Indexing pipeline: turns cleaned documents (data/processed/fastapi_docs.json)
into overlapping chunks, embeds them with a local sentence-transformers
model, and stores them in a persistent ChromaDB collection.

Also writes data/processed/chunks.json — the same chunk records ChromaDB
stores, kept as a plain file so retrieval.py can build a BM25 keyword index
(stage 3) without having to read them back out of the vector store.
"""

import json
import os
import re
from pathlib import Path

# See retrieval.py for why this must be set before importing transformers.
os.environ.setdefault("USE_TF", "0")

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

# Baseline embedding model: small, fast, no API key required. This is a
# deliberate starting point for the failure-analysis stage later — if
# retrieval recall turns out weak on technical/code-heavy queries, swapping
# this for a larger or retrieval-tuned model (e.g. BAAI/bge-small-en-v1.5)
# is one of the first improvements to try.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "fastapi_docs"

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def split_into_sections(content):
    """Split a document's markdown into (heading, section_text) pairs so
    each chunk can carry its nearest heading as context. Text before the
    first heading (should be none, since ingestion keeps the H1 first) is
    yielded with heading=None."""
    matches = list(HEADING_RE.finditer(content))
    if not matches:
        return [(None, content)]

    sections = []
    if matches[0].start() > 0:
        sections.append((None, content[: matches[0].start()]))

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections.append((heading, content[start:end]))

    return sections


def chunk_text_by_tokens(tokenizer, text, chunk_size, overlap):
    """Slide a token-sized window over `text` and return the exact
    substrings covered by each window, using the tokenizer's character
    offsets rather than decode() — decode() re-joins sub-word pieces with
    single spaces and destroys code indentation/newlines, which matters a
    lot for a code-heavy technical doc set."""
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    if not offsets:
        return []

    pieces = []
    step = chunk_size - overlap
    total_tokens = len(offsets)
    start = 0
    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        pieces.append(text[char_start:char_end])
        if end >= total_tokens:
            break
        start += step
    return pieces


def build_chunks(docs, tokenizer):
    chunks = []
    for doc in docs:
        for heading, section_text in split_into_sections(doc["content"]):
            section_text = section_text.strip()
            if not section_text:
                continue

            for piece in chunk_text_by_tokens(tokenizer, section_text, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS):
                piece = piece.strip()
                if not piece:
                    continue

                if heading and heading != doc["title"]:
                    heading_prefix = f"{doc['title']} > {heading}"
                else:
                    heading_prefix = doc["title"]

                chunks.append({
                    "chunk_id": f"{doc['id']}-{len(chunks)}",
                    "doc_id": doc["id"],
                    "doc_title": doc["title"],
                    "section": doc["section"],
                    "path": doc["path"],
                    "heading": heading,
                    "text": f"{heading_prefix}\n\n{piece}",
                })
    return chunks


def embed_and_store(chunks, embedder):
    texts = [c["text"] for c in chunks]
    print(f"Computing embeddings for {len(texts)} chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64).tolist()

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing_names = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        COLLECTION_NAME, metadata={"embedding_model": EMBEDDING_MODEL_NAME}
    )

    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "doc_id": c["doc_id"],
            "doc_title": c["doc_title"],
            "section": c["section"],
            "path": c["path"],
            "heading": c["heading"] or "",
        }
        for c in chunks
    ]

    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            documents=texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
        print(f"Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks into Chroma")

    return collection


def main():
    docs_path = PROCESSED_DIR / "fastapi_docs.json"
    if not docs_path.exists():
        raise SystemExit(f"{docs_path} not found -- run src/ingestion.py first.")

    docs = json.loads(docs_path.read_text(encoding="utf-8"))

    print(f"Loading tokenizer and embedding model: {EMBEDDING_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chunks = build_chunks(docs, tokenizer)
    print(f"Built {len(chunks)} chunks from {len(docs)} documents")

    chunks_path = PROCESSED_DIR / "chunks.json"
    chunks_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved chunk records -> {chunks_path}")

    embed_and_store(chunks, embedder)
    print(f"\nDone. Chroma collection '{COLLECTION_NAME}' persisted at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
