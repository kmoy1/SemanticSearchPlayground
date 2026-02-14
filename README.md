# Semantic Search Playground (Grounded RAG)

A Retrieval-Augmented Generation (RAG) app built with Streamlit, LangChain, FAISS.

NOTE: OPENAI_API_KEY needs to be configured- this does not run an LLM until it does, and will only fetch the closest semantically-relevant document without an LLM answer.

## Features

- Upload `.txt` documents
- Chunk documents and build a FAISS vector index
- Retrieve top-k relevant chunks for each question
- Generate grounded answers using an OpenAI model
- Show citations (source file + chunk id)
- Cache query results per index configuration

## Project Structure

- `app.py`: Entry point
- `app_core/ui.py`: UI logic + component definitions
- `app_core/state.py`: Session state (e.g. training corpus + trained index), cache keys
- `app_core/settings.py`: defaults + other configs
- `app_core/rag.py`: Logic for doc chunking, retrieval (FAISS vector store usage), and LLM generation (calling OpenAI LLM)

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

## Testing

Use `test_docs/` for testing:

1. Upload all files in `test_docs/`
2. Click `Train/Rebuild Index`
3. Run queries:
   - `How many PTO days do full-time employees get?` (Should return `benefits.txt`)
   - `When does health insurance start for new employees?` (Should return `company_policy.txt`)
   - `What team offsite ideas are listed?` (Should return `random_notes.txt`)
