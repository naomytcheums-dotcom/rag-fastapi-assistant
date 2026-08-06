"""
Chat dashboard for the FastAPI documentation RAG assistant.

Run with: streamlit run dashboard/app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from generation import Generator  # noqa: E402
from retrieval import CROSS_ENCODER_MODEL_NAME, EMBEDDING_MODEL_NAME, FINAL_TOP_K, Retriever  # noqa: E402

SUGGESTIONS = {
    ":material/http: How do I handle a 404 error?": "How do I handle a 404 error in FastAPI?",
    ":material/lock: How does OAuth2 + JWT auth work?": "How do I implement OAuth2 password flow with JWT tokens?",
    ":material/bolt: async def vs def?": "What is the difference between async def and def in FastAPI?",
    ":material/link: How do I inject a dependency?": "How do I declare a reusable dependency with Depends?",
}


st.set_page_config(
    page_title="FastAPI docs assistant",
    page_icon=":material/data_object:",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading retriever (embeddings + BM25 + reranker)...")
def get_retriever():
    return Retriever()


@st.cache_resource(show_spinner=False)
def get_generator():
    try:
        return Generator()
    except EnvironmentError:
        return None


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.markdown("### FastAPI docs assistant")
    st.caption("Answers grounded in the official FastAPI documentation, with citations.")

    if st.button("New conversation", icon=":material/add:", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    dev_mode = st.toggle("Developer mode", value=False, help="Show retrieval/generation internals and metrics")

    st.markdown("---")
    st.caption(f"Embedding model: `{EMBEDDING_MODEL_NAME}`")
    st.caption(f"Reranker: `{CROSS_ENCODER_MODEL_NAME}`")
    st.caption(f"Top-k retrieved: {FINAL_TOP_K}")


def render_sources(sources, dev_mode):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", icon=":material/description:"):
        for s in sources:
            label = s["heading"] or s["doc_title"]
            st.markdown(f"**[{s['index']}]** `{s['path']}` — {label}")
            if dev_mode and "rerank_score" in s:
                st.caption(f"rerank score: {s['rerank_score']:.4f}")


def render_metrics(metrics):
    if not metrics:
        return
    cols = st.columns(3)
    cols[0].metric("Retrieval", f"{metrics['retrieval_ms']} ms")
    if metrics.get("generation_ms") is not None:
        cols[1].metric("Generation", f"{metrics['generation_ms']} ms")
    cols[2].metric("Chunks used", metrics["n_chunks"])


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("error"):
                st.error(msg["error"], icon=":material/error:")
            render_sources(msg.get("sources"), dev_mode)
            if dev_mode:
                render_metrics(msg.get("metrics"))


prompt = None
if not st.session_state.messages:
    st.markdown("#### Ask anything about FastAPI")
    st.caption("Try one of these, or type your own question below.")
    selected = st.pills("Suggestions", list(SUGGESTIONS.keys()), label_visibility="collapsed")
    if selected:
        prompt = SUGGESTIONS[selected]

typed = st.chat_input("Ask a question about FastAPI")
if typed:
    prompt = typed

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        error_message = None
        answer_text = ""
        sources = []
        metrics = None

        with st.status("Searching the docs...", expanded=False) as status:
            try:
                retriever = get_retriever()
                t0 = time.perf_counter()
                chunks = retriever.retrieve(prompt, top_k=FINAL_TOP_K)
                retrieval_ms = round((time.perf_counter() - t0) * 1000)
            except (RuntimeError, FileNotFoundError, ValueError) as exc:
                status.update(label="Failed", state="error")
                error_message = f"Retrieval failed: {exc}"
                chunks = None

            if chunks is not None:
                status.update(label="Generating answer...")

                generator = get_generator()
                generation_ms = None

                if generator is None:
                    error_message = (
                        "ANTHROPIC_API_KEY is not configured, so I can't generate an answer -- "
                        "but retrieval still ran, see the sources below."
                    )
                else:
                    try:
                        t1 = time.perf_counter()
                        result = generator.generate(prompt, chunks)
                        generation_ms = round((time.perf_counter() - t1) * 1000)
                        answer_text = result["answer"]
                    except Exception as exc:
                        error_message = f"Generation failed: {exc}"

                sources = [
                    {
                        "index": i,
                        "path": c["path"],
                        "heading": c["heading"],
                        "doc_title": c["doc_title"],
                        "rerank_score": c["rerank_score"],
                    }
                    for i, c in enumerate(chunks, start=1)
                ]
                metrics = {"retrieval_ms": retrieval_ms, "generation_ms": generation_ms, "n_chunks": len(chunks)}

                status.update(label="Done", state="complete")

        if answer_text:
            st.write(answer_text)
        if error_message:
            st.error(error_message, icon=":material/error:")

        render_sources(sources, dev_mode)
        if dev_mode:
            render_metrics(metrics)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources,
        "metrics": metrics,
        "error": error_message,
    })
