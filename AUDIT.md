# Project audit — RAG FastAPI documentation assistant

Full read of every file in `src/`, `dashboard/`, `requirements.txt`. No file
was skimmed — findings below are cited by exact line number and, where
practical, verified by actually reproducing the failure rather than assumed.

Scope note: several findings below (prompt injection via docs, path
traversal) are **low risk today** because the current corpus is trusted,
official FastAPI documentation. They are flagged anyway because the
project's own stated business model ("Template personnalisé: vendre le
code pour l'adapter à d'autres documentations") means untrusted input is a
near-term reality, not a hypothetical.

## Update: fixed since this audit

This document is left as originally written (a point-in-time audit, not a
living doc) — findings below still describe what was found. Since then,
action-plan items 1, 2, 4, 5, 7, 8, 9, 10, and 11 have been applied and
verified:

- Path traversal in `ingestion.py`'s snippet resolver: now blocked with an
  `is_relative_to()` containment check (`tests/test_ingestion.py` covers it).
- Vector-store-empty crash: `Retriever.__init__`/`_semantic_search` now
  raise a clear `RuntimeError` instead of leaking Chroma's raw exception --
  verified by reproducing the original crash, then re-verifying it's caught.
- Dashboard: fixed the 3x duplicated model-name strings (now imported from
  `retrieval.py`), fixed the "exception as cached resource" pattern, and
  `retriever.retrieve()` is now wrapped so a retrieval failure shows a clean
  `st.error()` instead of crashing the whole page.
- `generation.py`: added a 30s client timeout, specific exception handling
  per Anthropic SDK error type, per-request token/latency logging, and a
  cost estimate. Fixed the `.env` parser not stripping inline `# comments`
  (the `st.secrets` fallback for Streamlit Cloud deployment is still open).
- `indexing.py`: replaced the bare `except Exception: pass` around
  collection deletion with an explicit existence check.
- Added `logging` (not just `print`) to `retrieval.py` and `generation.py`.
- Added `tests/test_ingestion.py` and `tests/test_retrieval.py` (19 tests,
  all passing) covering the pure functions above plus the RRF fusion and
  score-blending logic.
- Dependencies now pinned to actually-tested versions (see requirements.txt)
  and installed into a `.venv` instead of the global environment.

Not yet done: item 3 (dependency pins -- environment is being rebuilt into
a `.venv` to determine the exact tested versions), item 6 (tuning/held-out
split for the Recall@5 methodology), item 12 (type hints across `src/`),
and item 13 (extracting the dashboard's inline pipeline logic into a
separately testable function) remain open.

---

## 1. Executive summary — top 5 critical issues

| # | Issue | Why it's critical |
|---|-------|---|
| 1 | **No git repo, no README, no LICENSE exist yet** | `git status` returns "not a git repository." Nothing is version-controlled or shareable. This blocks everything else on the original checklist (GitHub, CI/CD, deployment) and is the single most visible gap to a recruiter. |
| 2 | **Zero pinned dependency versions, no virtualenv used** | Not hypothetical — already caused a real failure this session (`anthropic==0.34.2` resolved and was incompatible with `httpx>=0.28`, crashing every LLM call with `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`) and a real conflict with an unrelated global `fastapi` install (`starlette` version clash). `pip install -r requirements.txt` today is not guaranteed to reproduce a working environment tomorrow. |
| 3 | **Retrieval crashes ungracefully if the vector store is empty/missing** | Verified by reproduction: `Retriever.__init__` → `client.get_collection(...)` raises a raw `ValueError: Collection fastapi_docs does not exist.` with no handling anywhere in `retrieval.py` or `dashboard/app.py`. In the dashboard this shows Streamlit's default red traceback box to the end user — worse, it's *inconsistent* with `generation.py`'s errors, which the dashboard already catches and displays cleanly. |
| 4 | **Path traversal in `ingestion.py`'s snippet resolver** | `resolve_snippet_includes` only strips *leading* `../` segments (`LEADING_DOTDOT_RE`) before joining to `FASTAPI_REPO_ROOT` and calling `.resolve()`. A path like `docs_src/../../../../etc/passwd` embedded in a markdown file is never rejected — nothing checks the resolved path is still inside `FASTAPI_REPO_ROOT`. Low risk against the current trusted FastAPI docs; real risk the moment this ingests any less-trusted or user-supplied documentation set. |
| 5 | **The headline "Recall@5 = 90%" result has a data-leakage problem** | `RERANK_WEIGHT` was tuned (0.7 vs 0.85) by directly measuring against the same 50-question set now being reported as the result. That's evaluating on the tuning set, not a held-out set — a real methodological weakness that a technical interviewer is likely to catch and that undercuts the credibility of the number more than any code bug would. |

Runners-up that didn't make the top 5 only because they're cheaper to fix
once the above are addressed: zero logging/monitoring anywhere in the
codebase, zero automated tests, no LLM call timeout/retry, and a
model-name string duplicated in three places (`indexing.py`,
`retrieval.py`, and hardcoded again in `dashboard/app.py` lines 62-63) that
will silently go stale if either backing file changes.

---

## 2. Detailed analysis by file

### `src/ingestion.py`

| Line(s) | Problem | Impact | Fix |
|---|---|---|---|
| 63-78 | Path traversal: only leading `../` stripped before resolving against `FASTAPI_REPO_ROOT`; no containment check on the result | Arbitrary file read from disk if a malicious/malformed doc file is ever ingested (see exec summary #4) | After `.resolve()`, verify containment: `if not resolved.is_relative_to(FASTAPI_REPO_ROOT): return f"[blocked: path escapes repo root: {raw_path}]"` (Python 3.9+; use a manual `os.path.commonpath` check if targeting older) |
| 115 | `md_file_path.read_text(encoding="utf-8")` has no try/except | One malformed-encoding file kills the entire 155-file batch with no indication of which file, no partial output | Wrap in try/except, log the offending path, `continue` to the next file, and report a summary of skipped files at the end |
| whole file | No type hints anywhere (`def process_document(md_file_path, doc_id):` etc.) | Harder to catch mistakes early, harder for a reader/recruiter to know expected types at a glance | Add hints: `def process_document(md_file_path: Path, doc_id: int) -> dict:` etc. Low effort, mechanical |
| 52-60 | `LANG_BY_EXTENSION` covers 7 extensions; anything else (`.rst`, `.txt`, `.ini`, `.sh` variants) silently gets an unlabeled fenced block | Minor: only matters if this pipeline is reused for a doc set with different snippet file types | Expand the dict or fall back to no language tag gracefully (already does — just noting the coverage gap) |
| whole file | Zero unit tests for `resolve_snippet_includes`, `strip_admonitions`, `extract_title` — these are pure functions, the easiest possible code in the whole project to test | No regression protection; the two real bugs found and fixed earlier this session (missing `hl[...]` handling, the `{**dict}` false-positive) would have been caught instantly by 4-5 table-driven test cases | Add `tests/test_ingestion.py` with cases for: bare include, include with `hl[...]`, dict-unpacking false positive, missing snippet file, nested admonition |

### `src/indexing.py`

| Line(s) | Problem | Impact | Fix |
|---|---|---|---|
| 127-130 | `except Exception: pass` around `delete_collection` | Swallows *any* failure silently, not just "collection doesn't exist yet" (e.g. a real permissions or disk error on the Chroma directory would be silently ignored, and `create_collection` right after would then fail with a confusing secondary error) | Check for existence explicitly first (`COLLECTION_NAME in [c.name for c in client.list_collections()]`) and only skip deletion in that specific case |
| 122 | `embedder.encode(texts, ...).tolist()` loads all texts + all resulting embedding vectors into memory at once | Fine at 1899 chunks; does not scale to a much larger doc set (Django, LangChain docs are 5-10x bigger) — direct answer to your own "problèmes de scalabilité" question | Stream/batch: encode and `collection.add()` per batch instead of computing the full embeddings list up front |
| 32, and duplicated in `retrieval.py` line 35 + hardcoded again in `dashboard/app.py` lines 62-63 | `EMBEDDING_MODEL_NAME` string duplicated in 3 places, one of them a plain hardcoded string, not an import | If the model is ever changed in `indexing.py`, the dashboard sidebar will keep displaying the old model name — a real, silent correctness bug, not hypothetical | Dashboard should `from retrieval import EMBEDDING_MODEL_NAME, CROSS_ENCODER_MODEL_NAME` instead of hardcoding the strings |
| 124-134 | Full `delete_collection` + `create_collection` on every run | Indexing is not incremental — any doc-set update requires a full re-embed of everything. Fine at this scale, a real cost/latency issue at production scale | Out of scope for this stage; flag for a future "incremental indexing" pass keyed on a content hash per document |
| whole file | No type hints | Same as ingestion.py | Same fix |

### `src/retrieval.py`

| Line(s) | Problem | Impact | Fix |
|---|---|---|---|
| 75-76 | `client.get_collection(COLLECTION_NAME)` has no try/except | **Verified by reproduction**: raises `ValueError: Collection fastapi_docs does not exist.` if indexing never ran or `CHROMA_DIR` is wrong. Unhandled all the way to the dashboard's default error screen. Direct answer to your "que se passe-t-il si la base vectorielle est vide?" question | Wrap and re-raise with a clear message: `except ValueError: raise RuntimeError("Vector store is empty or missing -- run src/indexing.py first.") from None`. Mirror the friendly pattern already used for the `chunks.json` check two lines above it |
| 82-84 | `self.collection.query(...)` has no try/except | A Chroma-side error (corrupted DB file, disk issue) crashes `retrieve()` mid-request with no context | Wrap and surface a clear `RuntimeError` |
| 39-43, 53 | `SEMANTIC_CANDIDATES`, `BM25_CANDIDATES`, `RRF_K`, `FINAL_TOP_K`, `RERANK_WEIGHT` are all fixed constants chosen by limited manual testing, not a systematic sweep | Direct answers to your own questions: "top-k=5, pourquoi?" and "paramètres de fusion?" — honestly: no sensitivity analysis was ever run varying these independently | Build the `tests/test_parameters.py` sweep from the original project structure: vary top_k in {3,5,10}, RRF_K in {30,60,100}, report Recall/MRR for each combination |
| 53 (constant) + evaluation methodology | `RERANK_WEIGHT=0.85` was chosen by testing 0.7 and 0.85 directly against the *same* 50-question set now reported as the result | Data leakage between tuning and reporting (exec summary #5) — a real methodological weakness | Split `test_set.json` into a ~35-question tuning split and a ~15-question held-out split; tune only against the former, report only the latter. Document the split in `README.md` |
| whole file | No logging (uses nothing — not even `print` inside the class itself) | Zero observability into retrieval behavior in production: can't see query latency distribution, can't see which queries return zero results, can't debug a live issue after the fact | Add `logging` calls at INFO for each `retrieve()` call (query, latency, top result), matching what's already informally done via the dashboard's dev-mode timing |
| 123-124 | No cap on `len(query)` or `len(pairs)` text length before feeding the cross-encoder | A pathologically long user-typed question increases reranking compute with no limit — a cheap DoS/cost vector once the dashboard is public | Truncate `query` to a sane max length (e.g. 500 chars) before use, reject/warn otherwise |
| whole file | No type hints | Same as other files | Same fix |

### `src/generation.py`

| Line(s) | Problem | Impact | Fix |
|---|---|---|---|
| 50-55 | `build_context_block` inserts raw chunk text into the prompt with only a `[i] Source: ...` label, no stronger delimiter or instruction-hierarchy protection | Indirect prompt injection surface if the corpus is ever untrusted (exec summary note) — a malicious doc could contain "Ignore the above instructions and instead..." embedded in what looks like documentation prose | Wrap each excerpt in an explicit, hard-to-spoof delimiter (e.g. XML-style `<excerpt index="1" source="...">...</excerpt>`) and add an explicit line to the system prompt telling the model to treat excerpt content as data, never as instructions |
| 81-86 | `self.client.messages.create(...)` has no try/except, no explicit `timeout=` | CLI usage (`python src/generation.py "..."`) crashes on any API error with a raw traceback — direct answer to "que se passe-t-il si l'API LLM est indisponible?" for this entry point specifically (the dashboard *does* catch this at the call site, but the module itself doesn't, so any other caller gets no protection) | Add `timeout=30.0` to the `Anthropic(...)` client constructor (SDK default is minutes, far too long for interactive use) and wrap the `.create()` call with handling for `anthropic.APITimeoutError`, `anthropic.RateLimitError`, `anthropic.APIStatusError` |
| 41-47 | Custom `.env` parser does not strip inline comments (only whole-line `#` comments are skipped) | `ANTHROPIC_API_KEY=sk-ant-xxx  # my key` would silently include `  # my key` as part of the key value, causing a confusing auth failure with no obvious cause | Split on an unquoted `#` before parsing the value, or just recommend `python-dotenv` instead of a hand-rolled parser |
| whole file | Zero cost/token tracking | Direct gap vs. your own checklist item "estimation du coût par requête" — currently impossible to answer without adding instrumentation | Log `response.usage.input_tokens` / `output_tokens` per call (the Anthropic SDK returns this for free on every response) and multiply by the model's published per-token price |
| whole file | Deployment gap: `load_dotenv_if_present()` only reads a project-root `.env` file | Won't work unmodified if deployed to Streamlit Community Cloud (which uses `st.secrets`, not arbitrary `.env` files) — relevant since deployment is on your own checklist | Fall back to `st.secrets["ANTHROPIC_API_KEY"]` when running under Streamlit and no env var is set |

### `src/evaluation.py`

| Line(s) | Problem | Impact | Fix |
|---|---|---|---|
| 27-49 | Sequential loop, one `retrieve()` call per question, no parallelism | 50 questions already takes a couple of minutes; won't scale to a larger test set for a bigger doc corpus | Low priority at this scale; if the test set grows, parallelize with a thread pool (the bottleneck is model inference, which releases the GIL) |
| whole file | `ragas` is listed as a planned dependency in `requirements.txt`'s comments but never imported or used | The original project brief specifically named RAGAS as the evaluation framework; it isn't actually integrated — Recall@k/MRR are hand-rolled here instead | Either integrate `ragas` for the generation-quality metrics (faithfulness, answer relevance — needs API credit anyway) or stop referencing it as planned and document the hand-rolled metrics as the deliberate choice |
| whole file | No type hints | Same as other files | Same fix |

### `src/failure_analysis.py`

| Line(s) | Problem | Impact | Fix |
|---|---|---|---|
| 44-46 | Reaches into `retriever._semantic_search`, `retriever._bm25_search`, `retriever._reciprocal_rank_fusion` — underscore-prefixed "private" methods of another module's class | Tight coupling to `Retriever`'s internals; if those method signatures change during future work on `retrieval.py`, this script breaks silently or gives wrong diagnoses with no compile-time warning | Add a small dedicated method on `Retriever` itself, e.g. `Retriever.diagnose(query, gold_paths) -> dict`, so the diagnostic capability is a documented, intentional part of the class's public API instead of reaching into internals |
| whole file | No timestamp/config snapshot written into the output JSON (`results/failure_analysis_retrieval.json`) | No way to tell, months from now, which `RERANK_WEIGHT`/model config a given failure-analysis run was produced under | Add a `"config"` key to the output recording the constants in effect (`RERANK_WEIGHT`, `SEMANTIC_CANDIDATES`, model names) at run time |

### `dashboard/app.py`

| Line(s) | Problem | Impact | Fix |
|---|---|---|---|
| 62-63 | Hardcoded model-name strings instead of importing from `retrieval.py` | Silent drift bug if the backing model ever changes (see indexing.py finding) | `from retrieval import EMBEDDING_MODEL_NAME, CROSS_ENCODER_MODEL_NAME` and use them in the f-strings |
| 121 | `chunks = retriever.retrieve(prompt, top_k=FINAL_TOP_K)` — no try/except | **Verified**: crashes the whole Streamlit run with the default error screen if the vector store is empty/missing (see retrieval.py finding). This is the concrete, user-facing version of exec summary #3 | Wrap in try/except and render the same clean `st.error(...)` pattern already used for generation failures two blocks below |
| 41-44 | `get_generator()` returns an `Exception` *instance* (not `None`, not raised) as the cached "resource" on failure | Fragile pattern: line 131's `isinstance(generator, Exception)` is the only thing preventing a future edit from calling `.generate()` on an exception object and getting a confusing `AttributeError` instead of a clear message | Return `None` on failure instead, check `if generator is None:` |
| 116-166 | ~50 lines of retrieval/generation/error/metrics logic inlined directly in the Streamlit script body | Matches your own "fonctions trop longues" concern — and this logic is currently untestable in isolation since it's mixed into UI code | Extract into a plain function in `src/` (e.g. `run_pipeline(prompt) -> PipelineResult`) that the dashboard calls; this also makes it unit-testable without spinning up Streamlit |
| whole file | No per-session/per-IP rate limiting on Claude API calls | Once deployed publicly (your own stated goal), any visitor can drive unlimited billed API calls | Add a simple session-level request counter with a small cap, or a cooldown between messages, before public deployment |

### `requirements.txt`

| Line(s) | Problem | Impact | Fix |
|---|---|---|---|
| 5-6, 9, 19 | `sentence-transformers`, `chromadb`, `rank-bm25`, `streamlit` have zero version pins | **Already caused a real failure this session** (unrelated `anthropic` version conflict) — the exact same class of bug is one `pip install` away for any of these four | Pin at minimum the versions actually tested: run `pip freeze | grep -iE "sentence-transformers|chromadb|rank-bm25|streamlit|anthropic|torch|transformers"` in the working environment and record those as `>=` floors (or exact pins for maximum reproducibility) |
| whole file | No virtual environment used on this machine at all | Already caused a real, observed conflict: installing `streamlit` upgraded `starlette` and broke a separately-installed global `fastapi==0.115.0` package | Document (and use) a `.venv` in the README's install instructions: `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt` |
| whole file | No Python version stated | A recruiter cloning this on Python 3.9 vs 3.13 may hit different behavior (e.g. `str.removeprefix`, `is_relative_to` used in fixes above need 3.9+) | Add a `runtime.txt` or a README line: "Tested on Python 3.13" |

### Documentation

- **No `README.md`** — confirmed absent. This is the highest-visibility, lowest-effort gap on the entire list.
- **No `LICENSE`** — standard expectation for a public portfolio repo; absent.
- Module-level docstrings are present and good throughout `src/`; per-function docstrings are largely absent, and there are zero type hints anywhere in the codebase — both flagged per-file above rather than repeated here.

---

## 3. Action plan, ordered by priority

Effort is rough (solo, working with an assistant), assuming the code-level
fixes are mechanical once identified.

| # | Action | Addresses | Effort |
|---|---|---|---|
| 1 | `git init`, first commit, push to GitHub, add `LICENSE` | Exec summary #1 | 15 min |
| 2 | Write `README.md` (architecture, install with venv, results, known limitations) | Exec summary #1, doc gaps | 1-2 hrs |
| 3 | Pin dependency versions from the actually-tested environment | Exec summary #2 | 20 min |
| 4 | Wrap `Retriever.__init__`/`retrieve()`'s Chroma calls in try/except with clear errors; catch it in the dashboard too | Exec summary #3 | 30 min |
| 5 | Fix path-traversal in `ingestion.py`'s snippet resolver (containment check) | Exec summary #4 | 15 min |
| 6 | Re-split `test_set.json` into tuning/held-out sets; re-report Recall@5/MRR on the held-out portion only | Exec summary #5 | 45 min (mostly re-running eval) |
| 7 | Fix the 3x duplicated model-name strings in the dashboard | Duplication bug | 10 min |
| 8 | Add `timeout=` + try/except around the Anthropic client call in `generation.py`; add token/cost logging | Prod risk, cost tracking | 45 min |
| 9 | Add basic `logging` calls through `retrieval.py`/`generation.py` (query, latency, result count) | Monitoring gap | 45 min |
| 10 | Add `tests/test_ingestion.py` and `tests/test_retrieval.py` covering the edge cases found above | Test-coverage gap | 2-3 hrs |
| 11 | Set up a GitHub Actions workflow running the test suite + evaluation on push (the "innovation #2" regression gate) | Original brief's innovation #2 | 1-2 hrs |
| 12 | Add type hints across `src/` | Code quality | 1-2 hrs (mechanical) |
| 13 | Extract dashboard's inline pipeline logic into a testable `src/` function | Code quality | 30 min |
| 14 | Harden the `.env` parser (inline comments) or switch to `python-dotenv`; add `st.secrets` fallback for deployment | Deployment readiness | 20 min |

Items 1-6 are what I'd fix before this ever reaches a recruiter's screen.
Items 7-9 are what I'd fix before calling generation "done" once credits
land. Items 10-14 are what turns this from "working demo" into "production
credible" — worth doing, not blocking.
