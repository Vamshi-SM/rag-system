# RAG System

A production-ready Retrieval-Augmented Generation pipeline: document ingestion
(PDF/DOCX/TXT/Markdown), chunking, embeddings via an internal SLLM (Qwen)
API, a SQLite + `sqlite-vec` vector store, retrieval, LLM-generated answers,
a FastAPI REST API, and a full test suite.

```
Documents  ->  Load  ->  Chunk  ->  Embed  ->  Store  ->  Retrieve  ->  Prompt  ->  LLM  ->  Answer
```

## Table of contents

- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Running ingestion](#running-ingestion)
- [Running a query (CLI)](#running-a-query-cli)
- [Running the API](#running-the-api)
- [API documentation](#api-documentation)
- [Example requests](#example-requests)
- [Testing](#testing)
- [Configuration reference](#configuration-reference)
- [Security notes](#security-notes)
- [Extending the system](#extending-the-system)

---

## Architecture

```
                         ┌───────────────────────────────────────────────────────┐
                         │                     Data sources                      │
                         │        PDF   ·   DOCX   ·   TXT   ·   Markdown        │
                         └───────────────────────────┬───────────────────────────┘
                                                     │
                                                     ▼
                         ┌───────────────────────────────────────────────────────┐
                         │   src/loaders/          Phase 2 - Document Loading    │
                         │   PDFLoader · DocxLoader · TxtLoader · MarkdownLoader  │
                         │   DocumentLoader (recursive dir scan + dispatch)      │
                         └───────────────────────────┬───────────────────────────┘
                                                     │ LoadedDocument
                                                     ▼
                         ┌───────────────────────────────────────────────────────┐
                         │   src/chunking/         Phase 3 - Chunking            │
                         │   DocumentChunker (token-aware, per-page for PDFs)    │
                         └───────────────────────────┬───────────────────────────┘
                                                     │ Chunk
                                                     ▼
                         ┌───────────────────────────────────────────────────────┐
                         │   src/embeddings/        Phase 4 - Embeddings         │
                         │   BaseEmbedding  ·  QwenEmbedding (SLLM API)          │
                         └───────────────────────────┬───────────────────────────┘
                                                     │ EmbeddedChunk
                                                     ▼
                         ┌───────────────────────────────────────────────────────┐
                         │   src/vectordb/          Phase 5 - Vector Database    │
                         │   SQLite + sqlite-vec  ·  chunks / chunks_vec /       │
                         │   document_registry (dedup)                          │
                         └───────────────────────────┬───────────────────────────┘
                                                     │
                     ┌───────────────────────────────┼───────────────────────────┐
                     │                               │                           │
                     ▼                               ▼                           ▼
        ┌───────────────────────┐   ┌───────────────────────────┐   ┌───────────────────────┐
        │ src/ingestion/         │   │ src/retrieval/             │   │ src/llm/               │
        │ Phase 6 - Ingestion    │   │ Phase 7 - Retrieval        │   │ Phase 8 - LLM client   │
        │ IngestionService       │   │ Retriever (search/         │   │ BaseLLM · SLLMChat     │
        │ (Load→Dedup→Chunk→     │   │ retrieve/retrieve_with_    │   │ (chat completions API) │
        │  Embed→Store)          │   │ scores, top_k, threshold)  │   │                        │
        └───────────┬────────────┘   └──────────────┬─────────────┘   └───────────┬────────────┘
                    │                               │                             │
                    │                               └──────────────┬──────────────┘
                    │                                              ▼
                    │                               ┌───────────────────────────────┐
                    │                               │ src/rag/    Phase 8 - RAG      │
                    │                               │ prompt.py · response.py ·      │
                    │                               │ rag_pipeline.py (RAGPipeline)  │
                    │                               └───────────────┬────────────────┘
                    │                                                │
                    ▼                                                ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │                    src/api/            Phase 9 - REST API              │
        │   main.py (FastAPI app + lifespan)  ·  routes.py (/ingest /query        │
        │   /health /stats)  ·  schemas.py (Pydantic v2)  ·  dependencies.py      │
        │   middleware.py (request logging)  ·  exceptions.py (error handlers)   │
        └───────────────────────────┬──────────────────────────────────────────┘
                                    │
                     ┌──────────────┼──────────────┐
                     ▼              ▼              ▼
              scripts/ingest.py scripts/query.py  uvicorn (HTTP clients)
                (CLI)             (CLI)
```

**Design principles carried through every phase:**

- Each phase is a thin, swappable module behind an abstract base class
  (`BaseLoader`, `BaseEmbedding`, `BaseLLM`) - no phase reimplements another's
  logic.
- `IngestionService` is the *single* implementation of the ingestion pipeline;
  both `scripts/ingest.py` (CLI) and `POST /ingest` (API) call into it.
- `RAGPipeline` is the *single* implementation of the query pipeline; both
  `scripts/query.py` (CLI) and `POST /query` (API) call into it.
- Every layer degrades gracefully: a bad file, a failed embedding batch, a
  down LLM, or an empty retrieval result never crashes the pipeline - it's
  logged and surfaced as a clear, structured response.

## Project structure

```
rag-system/
├── data/
│   ├── documents/              # Drop source files here for ingestion
│   └── processed/               # SQLite database lives here
├── src/
│   ├── loaders/                 # Phase 2 - PDF/DOCX/TXT/MD loaders
│   ├── chunking/                 # Phase 3 - token-aware chunking
│   ├── embeddings/                # Phase 4 - SLLM (Qwen) embedding client
│   ├── vectordb/                   # Phase 5 - SQLite + sqlite-vec store
│   ├── ingestion/                   # Phase 6 - shared ingestion service
│   ├── retrieval/                    # Phase 7 - Retriever
│   ├── rag/                           # Phase 8 - prompt/response/pipeline
│   ├── llm/                            # Phase 8 - SLLM chat client
│   ├── api/                             # Phase 9 - FastAPI REST API
│   ├── utils/                            # Shared logging
│   └── config.py                          # Central settings (.env-driven)
├── scripts/
│   ├── ingest.py                 # CLI: run ingestion
│   └── query.py                  # CLI: interactive / one-shot query
├── tests/                         # Phase 10 - full test suite
├── .env.example
├── pytest.ini
├── requirements.txt
└── README.md
```

## Installation

Requires Python 3.11+.

```bash
git clone <this-repo>
cd rag-system
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your team's SLLM credentials:

```
SLLM_API_KEY=your-api-key
SLLM_BASE_URL=https://your-sllm-host/v1
EMBEDDING_MODEL=qwen-embedding-v1
CHAT_MODEL=qwen-chat-v1
```

> **API shape assumption:** the embedding and chat clients
> (`src/embeddings/qwen_embedding.py`, `src/llm/sllm.py`) assume
> OpenAI-compatible `POST {SLLM_BASE_URL}/embeddings` and
> `POST {SLLM_BASE_URL}/chat/completions` endpoints. If your team's actual
> SLLM API differs, only the `_call_api` method in each file needs to change.

## Running ingestion

Drop `.pdf`, `.docx`, `.txt`, or `.md` files (any nested folder structure)
into `data/documents/`, then:

```bash
python scripts/ingest.py
```

```
Loading documents...
15 documents loaded

Chunking...
352 chunks created

Generating embeddings...
352 embeddings generated

Saving vectors...
352 vectors stored

Total vectors in store: 352
Completed successfully.
```

Ingestion is **incremental and idempotent** - every file is checksummed
(SHA-256); re-running only processes files that are new or changed:

```
Loading documents...
15 documents loaded

15 duplicate document(s) skipped (already ingested)

Completed successfully. (no new documents)
```

Flags: `--data-dir <path>` to ingest a different directory, `--reset` to wipe
all vectors and the document registry first.

## Running a query (CLI)

Interactive mode:

```bash
python scripts/query.py
```

```
RAG Query Console. Type 'exit' or 'quit' to leave.

Ask a question

> How does the embedding system work?

Searching...
Retrieved 5 chunks.
Generating answer...

Answer

The embedding system first converts documents into chunks...

Sources

pricing.md
support.txt (page 2)

(1.84s, 5 chunks used)
```

One-shot mode:

```bash
python scripts/query.py "What is our refund policy?" --top-k 3 --threshold 0.75
```

## Running the API

```bash
uvicorn src.api.main:app --reload
```

Swagger UI at `http://localhost:8000/docs`, ReDoc at `http://localhost:8000/redoc`.

All expensive objects (embedding client, LLM client, vector store,
retriever, RAG pipeline, ingestion service) are constructed once at startup
(FastAPI `lifespan`) and reused across requests.

## API documentation

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Scan a folder, chunk/embed/store new documents, skip duplicates. |
| `POST` | `/query` | Answer a question via retrieval-augmented generation. |
| `GET` | `/health` | Check database, embedding service, and LLM connectivity. |
| `GET` | `/stats` | Document/chunk/embedding counts and database size on disk. |

All endpoints return `422` for validation errors and a structured error body
on any failure:

```json
{ "status": "error", "message": "Folder not found." }
```

`POST /ingest` additionally returns `403` if the requested folder resolves
outside the configured `ALLOWED_INGEST_ROOT` (see
[Security notes](#security-notes)), and `404` if the folder doesn't exist or
contains no supported documents.

## Example requests

### `POST /ingest`

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"folder": "./data/documents"}'
```

```json
{
    "status": "success",
    "documents": 12,
    "chunks": 415,
    "stored_vectors": 415,
    "duplicates_skipped": 0,
    "processing_time": "4.8 sec"
}
```

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does Council GPT perform embeddings?"}'
```

```json
{
    "answer": "Council GPT performs embeddings using the Qwen embedding model...",
    "sources": [
        {"filename": "doc1.pdf", "page": 2, "score": 0.94},
        {"filename": "doc3.pdf", "page": 8, "score": 0.89}
    ],
    "retrieved_chunks": 5,
    "latency": "1.42 sec"
}
```

Optional fields: `top_k` (1-50), `similarity_threshold` (-1.0-1.0), `filename`
(restrict retrieval to one source file).

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
    "status": "healthy",
    "database": "connected",
    "embedding_service": "online",
    "llm": "online"
}
```

### `GET /stats`

```bash
curl http://localhost:8000/stats
```

```json
{
    "documents": 24,
    "chunks": 925,
    "embeddings": 925,
    "database_size": "18.0 MB"
}
```

## Testing

```bash
pip install -r requirements.txt   # includes pytest, pytest-asyncio, httpx, reportlab
pytest
```

```bash
pytest -v                          # verbose
pytest tests/test_api.py -v        # one file
pytest tests/test_performance.py -v -s   # see benchmark report output
```

The suite runs **fully offline** - the embedding and LLM APIs are patched to
deterministic fakes (see `tests/conftest.py`), and PDF/DOCX samples are
generated on the fly (via `reportlab` / `python-docx`), so no network access
or real SLLM credentials are required to run it.

| File | Covers |
|---|---|
| `test_loaders.py` | Phase 2 - PDF/DOCX/TXT/Markdown loaders, directory recursion, unsupported/empty file handling |
| `test_chunking.py` | Phase 3 - chunk sizing, overlap, per-page metadata, empty documents |
| `test_embeddings.py` | Phase 4 - batching, retries, dimension detection, partial-batch failure handling |
| `test_vector_store.py` | Phase 5 - CRUD, cosine similarity search, document registry, persistence across reconnects |
| `test_ingestion.py` | Phase 6 - full ingestion service, duplicate detection, incremental ingestion, error capture |
| `test_retrieval.py` | Phase 7 - top-k, similarity threshold, metadata/filename filtering, error propagation |
| `test_rag_pipeline.py` | Phase 8 - prompt building, source de-duplication, graceful degradation on any failure |
| `test_llm.py` | Phase 8 - chat/generate/stream, retries, error handling |
| `test_api.py` | Phase 9 - all 4 endpoints, validation, error codes, Swagger docs availability |
| `test_integration.py` | End-to-end: 10 synthetic PDFs -> ingest -> query -> retrieve -> answer |
| `test_performance.py` | Ingestion throughput, embedding/retrieval latency, API response time (prints a benchmark report) |

## Configuration reference

All settings are read from `.env` (see `.env.example` for the full list with
defaults):

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIRECTORY` | `data/documents` | Default folder ingestion scans |
| `DATABASE_PATH` | `data/processed/vectors.db` | SQLite database file |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `700` / `100` | Chunking parameters (approx. tokens) |
| `SLLM_API_KEY` / `SLLM_BASE_URL` | - | SLLM credentials, shared by embeddings and chat |
| `EMBEDDING_MODEL` / `CHAT_MODEL` | - | Model names for embeddings vs. chat |
| `EMBEDDING_BATCH_SIZE` / `EMBEDDING_TIMEOUT` / `EMBEDDING_MAX_RETRIES` | `32` / `30` / `3` | Embedding client tuning |
| `REQUEST_TIMEOUT` / `LLM_MAX_RETRIES` | `60` / `3` | Chat client tuning |
| `TOP_K` / `SIMILARITY_THRESHOLD` | `5` / `0.7` | Default retrieval parameters |
| `RETRIEVAL_CANDIDATE_MULTIPLIER` | `4` | Over-fetch factor before metadata/filename filtering |
| `ALLOWED_INGEST_ROOT` | parent of `DATA_DIRECTORY` | Filesystem boundary `POST /ingest` may read from |
| `ALLOWED_CORS_ORIGINS` | (empty) | Comma-separated CORS origins; blank disables cross-origin access |
| `LOG_LEVEL` / `LOG_DIRECTORY` | `INFO` / `logs` | Logging configuration |

## Security notes

- **Path traversal**: `POST /ingest` accepts a folder path from the caller.
  To prevent it from being pointed at arbitrary filesystem locations (`/etc`,
  `~/.ssh`, etc.), the resolved path must fall inside `ALLOWED_INGEST_ROOT`
  (defaults to the parent of `DATA_DIRECTORY`). Requests outside that root
  get `403 Forbidden`.
- **CORS**: disabled by default (`ALLOWED_CORS_ORIGINS` empty). Set explicit
  origins for browser-based clients; avoid `*` with `allow_credentials=True`
  in production.
- **Secrets**: `SLLM_API_KEY` is read from `.env` only, never logged (only a
  warning is logged if it's missing, never its value).
- **Centralized error handling**: every error path returns the same
  `{"status": "error", "message": "..."}` shape - internal exception details
  (stack traces, file paths) are logged server-side, never leaked to clients
  (see the generic `Exception` handler in `src/api/exceptions.py`).

## Extending the system

- **New document type**: add a `BaseLoader` subclass in `src/loaders/`,
  register it in `DocumentLoader.__init__`.
- **New embedding/chat backend**: implement `BaseEmbedding` or `BaseLLM` and
  swap the import in `scripts/*.py` / `src/api/main.py`.
- **Hybrid search / re-ranking**: `Retriever.retrieve_with_scores` already
  over-fetches candidates and filters in Python - add a re-ranking step
  there without touching the vector store.
- **New vector DB backend**: implement the same public methods as
  `VectorStore` and swap the import.
