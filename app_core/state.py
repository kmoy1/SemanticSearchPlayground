import hashlib
from typing import Dict

import streamlit as st


def init_session_state() -> None:
    """Initialize Streamlit state keys needed for the app (Streamlit UI reads them)
    """
    if "documents" not in st.session_state:
        st.session_state.documents = {}
    if "query_cache" not in st.session_state:
        st.session_state.query_cache = {}


def reset_index_state() -> None:
    """Clear index (FAISS store) so we force retraining"""
    for key in ["faiss_store", "chunks", "index_signature", "query_cache"]:
        st.session_state.pop(key, None)


def build_index_signature(
    documents: Dict[str, str], chunk_size: int, chunk_overlap: int, chunk_count: int
) -> str:
    payload = "|".join(sorted(documents.keys())) + f"|{chunk_size}|{chunk_overlap}|{chunk_count}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_query_cache_key(
    query: str,
    retrieval_k: int,
    llm_provider: str,
    model_name: str,
    temperature: float,
    index_signature: str,
) -> str:
    payload = (
        f"{query}|{retrieval_k}|{llm_provider}|{model_name}|{temperature}|{index_signature}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
