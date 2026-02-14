# Semantic Search Playground (Grounded RAG)

A local-first Retrieval-Augmented Generation (RAG) app built with Streamlit, LangChain, and FAISS.

## Features

- Upload `.txt` documents
- Chunk documents and build a FAISS vector index
- Retrieve top-k relevant chunks for each question
- Generate grounded answers using an OpenAI model
- Show citations (source file + chunk id)
- Track token usage and estimated cost
- Cache query results per index configuration

## Project Structure

- `app.py`: thin Streamlit entrypoint
- `app_core/ui.py`: sidebar + page rendering logic
- `app_core/state.py`: session state initialization, cache keys, usage accumulation
- `app_core/settings.py`: defaults, pricing table, environment config helpers
- `app_core/rag.py`: document chunking, retrieval, and grounded LLM generation

## Setup

Use Python 3.11 or 3.12 (recommended).

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Optional (for LLM answer generation):

```bash
export OPENAI_API_KEY="your_api_key"
```

If `OPENAI_API_KEY` is not set, the app runs in retrieval-only mode.

## Run

```bash
streamlit run app.py
```

## App Flow

1. Upload text files
2. Build/rebuild index (chunking + embeddings + FAISS)
3. Ask a question
4. Review grounded answer and citations
