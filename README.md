# FastAPI docs assistant — a self-diagnosing RAG system

A retrieval-augmented Q&A assistant over the official FastAPI documentation
that answers developer questions with citations — and, unlike most RAG
demos, measures and documents *why* it fails and what fixing it actually
did to the numbers, instead of reporting a single feel-good metric.

**Status:** retrieval pipeline complete and evaluated. Generation is
implemented but blocked on Anthropic API credit — see
[Current limitations](#current-limitations).

## Why this project is different

Most portfolio RAG projects stop at "it works, here's a demo." This one
includes an honest audit trail:

- [`results/failure_analysis.md`](results/failure_analysis.md) — diagnosed
  *why* 6 of 50 test questions missed their target document (spoiler: 5 of
  6 weren't a search problem at all, the reranker was demoting correct
  results), applied a fix, and reports two iterations of that fix — the
  first one looked reasonable and actually made MRR worse. The second
  worked. Both are documented.
- [`AUDIT.md`](AUDIT.md) — a full security/quality/production-readiness
  audit of this exact codebase, including a finding that the reported
  Recall@5 number involved tuning a hyperparameter against the same set
  it's evaluated on (a real methodological weakness, called out rather
  than hidden).

## Results

| metric | before | after |
|---|:---:|:---:|
| Recall@5 | 88.0% | **90.0%** |
| MRR | 0.792 | 0.792-0.796* |

\* small run-to-run floating-point jitter in the cross-encoder means MRR
isn't perfectly reproducible bit-for-bit; see `AUDIT.md`. Recall@5 is
stable across runs.

Measured against a 50-question hand-labeled test set
(`data/test_set.json`, 20 easy / 20 medium / 10 hard, covering routing,
parameters, Pydantic validation, dependencies, security, WebSockets,
background tasks, testing, deployment, and more).

## Architecture

```
data/raw/*.md  (FastAPI docs, gitignored -- see Setup)
      |
      v
src/ingestion.py        clean markdown, resolve code-snippet includes,
                         strip HTML/admonitions
      |
      v
data/processed/fastapi_docs.json   (155 cleaned documents, committed)
      |
      v
src/indexing.py         chunk by real tokenizer offsets (512 tokens,
                         50 overlap), embed, store in ChromaDB
      |
      v
data/chroma/  (vector store, gitignored, rebuilt by indexing.py)
      |
      v
src/retrieval.py        hybrid search: BM25 + semantic, fused with
                         Reciprocal Rank Fusion, reranked with a
                         cross-encoder, blended with the fusion score
      |
      v
src/generation.py       Claude API call with a strict citation prompt
      |
      v
dashboard/app.py        Streamlit chat UI with sources + dev-mode metrics
```

Parallel pipelines: `src/evaluation.py` (Recall@k/MRR against the test
set), `src/failure_analysis.py` (diagnoses *why* a retrieval miss
happened), `tests/test_regression.py` (blocks a change if it drops a
metric >5% relative to `results/regression_baseline.json`), `src/llm_judge.py`
and `src/hallucination_detection.py` (faithfulness/relevance scoring and
claim-level hallucination checks — written, not yet verified against a
live call, see [Current limitations](#current-limitations)).

## Tech stack

| Stage | Choice | Why |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Fast, free, local baseline — deliberately not the fanciest option, see limitations |
| Vector store | ChromaDB (local, persistent) | Zero-config, no external service required |
| Keyword search | BM25 (`rank-bm25`) | Catches exact identifiers (`HTTPException`, `async def`) that embeddings can blur |
| Fusion | Reciprocal Rank Fusion | Combines two differently-scaled rankings without hand-tuned weights |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2`, blended 85/15 with the fusion score | See `results/failure_analysis.md` for why the blend exists |
| Generation | Claude API (`claude-sonnet-5`) | Strict citation-required system prompt |
| Dashboard | Streamlit | Chat UI, source citations, dev-mode metrics |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

The FastAPI documentation itself is not committed (`data/raw/` is
gitignored — it's a straight copy of another repo's docs). To rebuild it
from scratch:

```bash
git clone https://github.com/fastapi/fastapi.git ../fastapi
cp -r ../fastapi/docs/en/docs/* data/raw/
python src/ingestion.py
```

`data/processed/fastapi_docs.json` (the cleaned output) *is* committed, so
you can skip straight to indexing if you don't need to re-derive it:

```bash
python src/indexing.py          # builds chunks + vector store
python src/evaluation.py        # Recall@5 / MRR against data/test_set.json
python src/failure_analysis.py  # diagnoses retrieval misses
```

To run the dashboard:

```bash
streamlit run dashboard/app.py
```

Generation requires an Anthropic API key:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
python src/generation.py "How do I handle a 404 error in FastAPI?"
```

## Current limitations

Documented in full in [`AUDIT.md`](AUDIT.md). The headline ones:

- **Generation, hallucination detection, and LLM-judge calibration are
  blocked on API credit.** The code is written and imports cleanly, but
  none of it has been exercised against a live response yet.
- **The Recall@5/MRR headline numbers have a data-leakage caveat**: the
  reranker blend weight was tuned against the same 50-question set now
  used to report the result. A tuning/held-out split is planned but not
  yet done.
- No automated tests beyond the regression gate yet; no CI for anything
  other than the retrieval regression check.

## Project structure

```
rag-fastapi-assistant/
├── data/
│   ├── raw/              FastAPI docs copy (gitignored)
│   ├── processed/        cleaned docs + chunks (committed)
│   └── test_set.json     50 labeled questions
├── src/                  pipeline stages (see Architecture)
├── dashboard/app.py       Streamlit UI
├── tests/test_regression.py
├── calibration/           LLM-judge calibration harness (see calibration/README.md)
├── results/               evaluation + failure-analysis output
├── .github/workflows/     CI regression check
├── AUDIT.md               full code/security/quality audit
└── requirements.txt
```
