import time
from typing import Dict, List, Tuple

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI


def load_embeddings_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )


def build_documents(documents_by_name: Dict[str, str]) -> List[Document]:
    return [
        Document(page_content=content, metadata={"file_name": name})
        for name, content in documents_by_name.items()
    ]


def chunk_documents(
    documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 120
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def create_faiss_store(documents: List[Document], embeddings_model: HuggingFaceEmbeddings) -> FAISS:
    return FAISS.from_documents(documents, embeddings_model)


def retrieve_top_k(faiss_store: FAISS, query: str, k: int = 4) -> List[Tuple[Document, float]]:
    return faiss_store.similarity_search_with_score(query, k=k)


def _build_grounding_prompt(query: str, retrieved_docs: List[Tuple[Document, float]]) -> str:
    context_blocks = []
    for doc, _score in retrieved_docs:
        source = f"{doc.metadata.get('file_name', 'unknown')}#chunk_{doc.metadata.get('chunk_id', 'na')}"
        context_blocks.append(f"[{source}]\n{doc.page_content}")

    context_text = "\n\n".join(context_blocks)
    return (
        "Use only the provided context to answer the user. "
        "If the context is insufficient, say you do not have enough information. "
        "At the end of your answer, include a line `Citations:` followed by source ids in square brackets.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context_text}"
    )


def generate_grounded_answer(
    query: str,
    retrieved_docs: List[Tuple[Document, float]],
    api_key: str,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_retries: int = 2,
) -> Dict:
    if not api_key:
        raise ValueError("Missing OpenAI API key.")

    client = OpenAI(api_key=api_key)
    prompt = _build_grounding_prompt(query, retrieved_docs)
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a grounded RAG assistant. Never use outside knowledge.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            usage = response.usage
            usage_dict = {
                "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            }
            return {
                "answer": response.choices[0].message.content,
                "usage": usage_dict,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                raise last_error
