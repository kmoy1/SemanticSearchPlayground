import hashlib
from typing import Dict

import streamlit as st


USAGE_DEFAULTS = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "estimated_cost_usd": 0.0,
}


def init_session_state() -> None:
    """Initialize Streamlit state keys needed for the app (Streamlit UI reads them)
    """
    if "documents" not in st.session_state:
        st.session_state.documents = {}
    if "query_cache" not in st.session_state:
        st.session_state.query_cache = {}
    if "usage_totals" not in st.session_state:
        st.session_state.usage_totals = USAGE_DEFAULTS.copy()


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
    model_name: str,
    temperature: float,
    index_signature: str,
) -> str:
    payload = f"{query}|{retrieval_k}|{model_name}|{temperature}|{index_signature}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def add_usage(usage: Dict[str, float]) -> None:
    st.session_state.usage_totals["input_tokens"] += usage["input_tokens"]
    st.session_state.usage_totals["output_tokens"] += usage["output_tokens"]
    st.session_state.usage_totals["total_tokens"] += usage["total_tokens"]
    st.session_state.usage_totals["estimated_cost_usd"] += usage["estimated_cost_usd"]
