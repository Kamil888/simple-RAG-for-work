# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does
A lightweight RAG (Retrieval-Augmented Generation) app. Users upload office documents (PDF, DOCX, PPTX), which are chunked, embedded locally, and stored in ChromaDB. They can then query the documents via Claude or GPT-4o and get answers with source citations (filename + page/slide number). Re-uploading a document replaces its embeddings, enabling dynamic context updates.

## Running Locally
```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows bash; or .venv\Scripts\activate.bat in CMD
pip install -r requirements.txt
cp .env.example .env            # fill in ANTHROPIC_API_KEY and/or OPENAI_API_KEY
streamlit run app.py
```

## Running Tests
```bash
pytest tests/
pytest tests/test_vector_store.py   # single file
pytest tests/ -k "re_upload"        # single test by name
```

Tests use `monkeypatch` to redirect ChromaDB to a temp directory — no real data is touched.

## Deployment (Streamlit Cloud)
Push to GitHub, connect at share.streamlit.io, set entry point `app.py`. Add API keys under **Settings > Secrets** in TOML format. Note: `data/chroma_db/` is ephemeral on Streamlit Cloud — users must re-upload documents after cold starts.

## Architecture

```
app.py                   ← Streamlit UI (sidebar upload + chat main area)
rag/
  ingestion/
    loader.py            ← dispatch by file extension → list of chunk dicts
    pdf_parser.py        ← pypdf, metadata: {source, page, chunk_index}
    docx_parser.py       ← python-docx, estimated page by cumulative char count
    pptx_parser.py       ← python-pptx, metadata: {source, slide, chunk_index}
  embedder.py            ← sentence-transformers all-MiniLM-L6-v2, loaded once via lru_cache
  vector_store.py        ← ChromaDB PersistentClient; add_document deletes then re-adds by filename
  retriever.py           ← embed query → vector_store.query → list of chunk dicts with score
  llm/
    base.py              ← abstract BaseLLM.complete(system, user) -> str
    claude_llm.py        ← claude-sonnet-4-6
    openai_llm.py        ← gpt-4o
  prompt.py              ← builds system prompt with [Source: file, page N] blocks
  pipeline.py            ← PipelineResult(answer, sources); orchestrates retrieve → prompt → LLM
data/chroma_db/          ← gitignored; persistent embeddings
```

## Key Design Decisions
- **Single ChromaDB collection** named `"documents"`. Each chunk has `metadata.source = filename`.
- **Re-upload = replace**: `add_document()` calls `collection.delete(where={"source": filename})` before inserting new chunks. Other files are untouched.
- **PPTX chunks** use `"slide"` key in metadata; PDF/DOCX use `"page"`. `pipeline.py` and `app.py` handle both.
- **Embedding model** (`all-MiniLM-L6-v2`) runs locally — no API key, no cost per query.
- **LLM is abstracted** behind `BaseLLM` so adding a new provider means creating one file in `rag/llm/`.
