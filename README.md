# Semantic Search Playground (Grounded RAG)

A Retrieval-Augmented Generation (RAG) app built with Streamlit, LangChain, and FAISS.

This app uses Ollama, a free local LLM, for answer generation.

## Features

- Upload `.txt` documents
- Chunk documents and build a FAISS vector index
- Retrieve top-k relevant chunks for each question
- Generate grounded answers using Ollama (local)
- Show citations (source file + chunk id)
- Cache query results per index configuration

## Project Structure

- `app.py`: Entry point
- `app_core/ui.py`: UI logic + component definitions
- `app_core/state.py`: Session state (corpus, index, cache keys)
- `app_core/settings.py`: Defaults + config helpers
- `app_core/rag.py`: Chunking, retrieval, and LLM generation

## Setup

Use Python 3.11 or 3.12 (recommended for Langchain compatibility).

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Ollama Setup (Required)
We need to set up and run Ollama separately before we can run this app. 

1. Install Ollama:

```bash
brew install ollama
```

2. Start Ollama:

```bash
ollama serve
```

3. In a second terminal, pull a model:

```bash
ollama pull llama3.2:3b
```

4. In the app sidebar, use:
   - `LLM Provider = ollama`
   - `LLM Model = llama3.2:3b`
   - `Ollama Base URL = http://localhost:11434`

## Run

```bash
streamlit run app.py
```

## Testing

Use `test_docs/` for testing:

1. Upload all files in `test_docs/`
2. Click `Train/Rebuild Index`
3. Run queries:
   - `How many PTO days do full-time employees get?` (Should return 20 days)
   - `When does health insurance start for new employees?` (Should return first day of the month after hire)
   - `What team offsite ideas are listed?` (Should list hiking trip, museum day, cooking class)
