# Contains UI + rendering logic for all app components.

from typing import Dict
from pathlib import Path

import streamlit as st

from app_core.rag import (
    build_documents,
    chunk_documents,
    create_faiss_store,
    generate_grounded_answer,
    load_embeddings_model,
    retrieve_top_k,
)
from app_core.settings import DEFAULTS, estimate_cost_usd, get_openai_api_key
from app_core.state import add_usage, build_index_signature, build_query_cache_key, reset_index_state


@st.cache_resource
def get_embeddings_model():
    return load_embeddings_model()


def _resolve_openai_api_key() -> str:
    """Return API key from env or Streamlit secrets without raising if secrets are missing."""
    env_key = get_openai_api_key()
    if env_key:
        return env_key

    secrets_paths = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
    ]
    if not any(path.exists() for path in secrets_paths):
        return ""

    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except FileNotFoundError:
        return ""


def render_sidebar() -> Dict:
    """Render pagenav options + RAG controls, then return selected values."""

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload Documents", "Query Engine"])

    st.sidebar.subheader("RAG Settings")
    settings = {
        "page": page,
        "chunk_size": st.sidebar.slider(
            "Chunk Size", min_value=300, max_value=1500, value=DEFAULTS.chunk_size, step=50
        ),
        "chunk_overlap": st.sidebar.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=400,
            value=DEFAULTS.chunk_overlap,
            step=20,
        ),
        "retrieval_k": st.sidebar.slider(
            "Top-K Retrieval", min_value=1, max_value=8, value=DEFAULTS.retrieval_k, step=1
        ),
        "model_name": st.sidebar.text_input("OpenAI Model", value=DEFAULTS.model_name),
        "temperature": st.sidebar.slider(
            "LLM Temperature", min_value=0.0, max_value=0.7, value=DEFAULTS.temperature, step=0.1
        ),
    }
    return settings


def render_upload_page() -> None:
    st.title("Upload Documents")
    st.write("Upload text documents to index for grounded Q&A.")

    # Create upload file component + store files in uploaded_files. Enforces only .txt files
    uploaded_files = st.file_uploader(
        "Upload .txt files",
        type=["txt"],
        accept_multiple_files=True,
    )

    # Upload files (or update file) logic
    if st.button("Add/Update Uploaded Files", use_container_width=True):
        if not uploaded_files:
            st.warning("Select at least one .txt file first.")
        else:
            changed = False
            for file in uploaded_files:
                content = file.read().decode("utf-8")
                if st.session_state.documents.get(file.name) != content:
                    changed = True
                st.session_state.documents[file.name] = content

            # Clear old index data so we can rebuild embeddings + FAISS index from uploaded docs
            if changed:
                reset_index_state()
                st.success("Documents updated. Index cache cleared; retrain required.")
            else:
                st.info("No content changes detected.")

    if st.button("Clear All Documents", use_container_width=True):
        st.session_state.documents = {}
        reset_index_state()
        st.success("All documents and index state cleared.")

    if st.session_state.documents:
        st.subheader("Current Documents")
        for doc_name, doc_content in st.session_state.documents.items():
            with st.expander(doc_name):
                st.write(doc_content)


def _render_usage(usage: Dict[str, float]) -> None:
    st.subheader("Usage")
    st.write(
        {
            "query_input_tokens": usage["input_tokens"],
            "query_output_tokens": usage["output_tokens"],
            "query_total_tokens": usage["total_tokens"],
            "query_estimated_cost_usd": round(usage["estimated_cost_usd"], 6),
            "session_total_tokens": st.session_state.usage_totals["total_tokens"],
            "session_estimated_cost_usd": round(
                st.session_state.usage_totals["estimated_cost_usd"], 6
            ),
        }
    )


def render_query_page(settings: Dict) -> None:
    st.title("Grounded RAG Q&A")

    if not st.session_state.documents:
        st.warning("No documents uploaded yet. Upload files first.")
        st.stop()

    st.subheader("Corpus")
    st.write(f"Loaded documents: {len(st.session_state.documents)}")

    if st.button("Train/Rebuild Index", use_container_width=True):
        with st.spinner("Chunking documents and building FAISS index..."):
            base_documents = build_documents(st.session_state.documents)
            chunks = chunk_documents(
                base_documents,
                chunk_size=settings["chunk_size"],
                chunk_overlap=settings["chunk_overlap"],
            )
            faiss_store = create_faiss_store(chunks, get_embeddings_model())
            index_signature = build_index_signature(
                st.session_state.documents,
                settings["chunk_size"],
                settings["chunk_overlap"],
                len(chunks),
            )

            st.session_state.faiss_store = faiss_store
            st.session_state.chunks = chunks
            st.session_state.index_signature = index_signature
            st.session_state.query_cache = {}
        st.success(f"Index ready with {len(chunks)} chunks.")

    if "faiss_store" not in st.session_state:
        st.info("Build the index to start asking questions.")
        st.stop()

    st.subheader("Ask a Question")
    with st.form("query_form"):
        query = st.text_input("Question")
        submit = st.form_submit_button("Run Grounded Query")

    if not (submit and query.strip()):
        return

    cache_key = build_query_cache_key(
        query,
        settings["retrieval_k"],
        settings["model_name"],
        settings["temperature"],
        st.session_state.get("index_signature", ""),
    )

    cached = st.session_state.query_cache.get(cache_key)
    if cached:
        result = cached
        st.caption("Served from query cache.")
    else:
        with st.spinner("Retrieving context..."):
            retrieved = retrieve_top_k(
                st.session_state.faiss_store,
                query,
                k=settings["retrieval_k"],
            )

        api_key = _resolve_openai_api_key()
        if api_key:
            with st.spinner("Generating grounded answer..."):
                generation = generate_grounded_answer(
                    query=query,
                    retrieved_docs=retrieved,
                    api_key=api_key,
                    model_name=settings["model_name"],
                    temperature=settings["temperature"],
                )
            answer_text = generation["answer"]
            usage = generation["usage"]
            usage["estimated_cost_usd"] = estimate_cost_usd(
                settings["model_name"],
                usage["input_tokens"],
                usage["output_tokens"],
            )
            add_usage(usage)
        else:
            answer_text = retrieved[0][0].page_content
            usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
            }

        result = {
            "answer": answer_text,
            "retrieved": retrieved,
            "usage": usage,
            "retrieval_only": not bool(api_key),
        }
        st.session_state.query_cache[cache_key] = result

    st.subheader("Answer")
    if result.get("retrieval_only"):
        st.warning(
            "OpenAI API key not configured, so this is retrieval-only mode. "
            "Set OPENAI_API_KEY to enable LLM grounded answers."
        )
        st.text_area("Retrieved Context", value=result["answer"], height=220, disabled=True)
    else:
        st.write(result["answer"])

    st.subheader("Citations")
    for rank, (doc, score) in enumerate(result["retrieved"], start=1):
        source = f"{doc.metadata.get('file_name', 'unknown')}#chunk_{doc.metadata.get('chunk_id', 'na')}"
        with st.expander(f"{rank}. {source} (distance: {score:.4f})"):
            st.write(doc.page_content)

    _render_usage(result["usage"])
