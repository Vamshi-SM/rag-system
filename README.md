# RAG System

Retrieval-Augmented Generation system for ingesting PDF, DOCX, TXT, and
Markdown documents, storing embeddings, retrieving relevant chunks, and
generating grounded answers.

The repository currently contains two backend paths:

- The CLI and Streamlit application use Google Vertex AI through
  `GoogleEmbedding` and `GeminiChat`.
- The FastAPI application is wired to the older `QwenEmbedding` and
  `SLLMChat` path and should be treated as a separate, unfinished integration
  until its credentials and constructor wiring are aligned with the active
  Google configuration.

The active `.env` selects PostgreSQL with pgvector. SQLite with sqlite-vec is
also supported when `DATABASE_BACKEND=sqlite`.

```text
Documents -> Load -> Chunk -> Embed -> Store -> Retrieve -> Prompt -> LLM -> Answer
```

## Project Structure

```text
rag-system/
├── data/
│   ├── documents/              # Source documents for the normal ingestion CLI
│   ├── upload_files/           # Files used by the watcher/Streamlit path
│   └── processed/              # SQLite database when the SQLite backend is used
├── src/
│   ├── loaders/                # PDF, DOCX, TXT, and Markdown loading
│   ├── chunking/               # Token-aware document chunking
│   ├── embeddings/             # Qwen/SLLM and Google Vertex AI embedders
│   ├── vectordb/               # Backend-neutral store, SQLite, and PostgreSQL
│   ├── ingestion/              # Shared Load -> Dedup -> Chunk -> Embed -> Store flow
│   ├── retrieval/              # Similarity and metadata-filtered retrieval
│   ├── rag/                    # Prompt construction and answer pipeline
│   ├── llm/                    # Gemini and compatibility wrappers
│   └── api/                    # FastAPI routes and application lifecycle
├── scripts/
│   ├── ingest.py               # Normal batch ingestion CLI
│   ├── query.py                # CLI query interface
│   ├── watcher.py              # Optional upload-folder watcher
│   └── app.py                  # Optional Streamlit UI
├── tests/                      # Offline unit and integration tests
├── .env.example                # Safe configuration template
├── requirements.txt
└── pytest.ini
```

## Installation

Use Python 3.11 or newer. On Windows, create a virtual environment so the
terminal and VS Code use the same interpreter:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Never commit `.env` or place API keys in source files. The current workspace
`.env` contains credentials; rotate any credential that has been exposed and
keep only placeholders in `.env.example`.

## Configuration

Configuration is loaded by `src/config.py` with `python-dotenv` when the
process starts. Paths are relative to the project root unless absolute paths
are supplied.

### Active Google + PostgreSQL configuration

The current CLI and UI configuration uses:

```dotenv
DATA_DIRECTORY=data/documents
DATABASE_BACKEND=postgres
DATABASE_URL=postgresql://user:password@host:5432/database
GCP_PROJECT_ID=your-gcp-project
GCP_LOCATION=asia-south1
EMBEDDING_MODEL=text-embedding-004
GEMINI_MODEL=gemini-2.5-flash
TOP_K=3
SIMILARITY_THRESHOLD=0.20
```

The PostgreSQL server must have the `vector` extension installed, and the
database user must be able to create the `chunks`, `document_registry`, and
vector index objects. The PostgreSQL schema currently uses `VECTOR(768)`;
the selected embedding model must produce 768-dimensional vectors.

Google authentication must also be available to the Google Gen AI client.
Use the Google Cloud SDK/application-default credentials or set
`GOOGLE_APPLICATION_CREDENTIALS` to a local service-account JSON path. Do not
hardcode that path in a script.

### SQLite alternative

For a local database, use:

```dotenv
DATABASE_BACKEND=sqlite
DATABASE_PATH=data/processed/vectors.db
```

SQLite requires no PostgreSQL server and uses `sqlite-vec`. The normal CLI
will create the configured database directory automatically.

### Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIRECTORY` | `data/documents` | Directory scanned by `scripts/ingest.py` |
| `DATABASE_BACKEND` | `sqlite` | `sqlite` or `postgres` |
| `DATABASE_PATH` | `data/processed/vectors.db` | SQLite database path |
| `DATABASE_URL` | empty | PostgreSQL connection URL |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `700` / `100` | Chunking parameters |
| `GCP_PROJECT_ID` / `GCP_LOCATION` | empty / `asia-south1` | Vertex AI project and region |
| `EMBEDDING_MODEL` | `text-embedding-004` | Embedding model |
| `EMBEDDING_BATCH_SIZE` | `32` | Batch size for configured embedding clients |
| `TOP_K` / `SIMILARITY_THRESHOLD` | `5` / `0.7` | Retrieval defaults |
| `RETRIEVAL_CANDIDATE_MULTIPLIER` | `4` | Candidate over-fetch factor |
| `CHAT_MODEL` | `ollama/kimi-k2.7-code` | Legacy SLLM chat setting |
| `REQUEST_TIMEOUT` / `LLM_MAX_RETRIES` | `60` / `3` | Legacy chat client settings |
| `ALLOWED_INGEST_ROOT` | parent of `DATA_DIRECTORY` | API ingestion path boundary |
| `ALLOWED_CORS_ORIGINS` | empty | Comma-separated CORS origins |
| `LOG_LEVEL` / `LOG_DIRECTORY` | `INFO` / `logs` | Logging configuration |

The Qwen/SLLM embedding client additionally reads `SLLM_API_KEY` or
`SHAREDLLM_API_KEY`, and `SLLM_BASE_URL`. Those values are used by the API
path, not by the current Google-based CLI ingestion path.

## Run Ingestion

Put supported files under `data/documents/`, then run from the project root:

```powershell
python scripts/ingest.py
```

Useful options:

```powershell
python scripts/ingest.py --help
python scripts/ingest.py --data-dir data/documents
python scripts/ingest.py --reset
```

The service recursively loads supported files, computes a SHA-256 checksum,
skips unchanged files, chunks new documents, calls the configured embedder,
and writes vectors to the selected database. `--reset` deletes stored vectors
and the document registry before ingesting again.

This command requires a reachable PostgreSQL database when
`DATABASE_BACKEND=postgres`, plus working Vertex AI credentials for the
current `scripts/ingest.py` implementation.

## Run Queries

One-shot query:

```powershell
python scripts/query.py "What is our refund policy?"
python scripts/query.py "What is our refund policy?" --top-k 3 --threshold 0.20
```

Interactive mode:

```powershell
python scripts/query.py
```

Type `exit` or `quit` to stop. The query CLI uses the same Google Vertex AI
embedding model and Gemini chat model as the current ingestion/UI path.

## Run the FastAPI API

```powershell
python -m uvicorn src.api.main:app --reload
```

Interactive API documentation is available at:

- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`

Routes are `POST /ingest`, `POST /query`, `POST /query/latency`, `GET /health`,
and `GET /stats`.
The ingest route only accepts folders inside `ALLOWED_INGEST_ROOT`.

The API wires its LLM from `SLLMChat` (currently aliased to `GeminiChat`,
constructed with `GCP_PROJECT_ID` / `GCP_LOCATION` / `GEMINI_MODEL`) and its
embedder from `EMBEDDING_BACKEND` (`ollama` by default — the local Ollama
`qwen3-embedding:0.6b` model; also supports `sharedllm` and `google`). Both
clients are constructed lazily on first use, so the server starts even before
credentials are configured — `/health` then reports those services as
`offline` while retrieval-only routes (`/stats`, `/query/latency`) keep working.

## Optional Upload Watcher and UI

The repository also contains `scripts/watcher.py` and `scripts/app.py` for an
automatic upload-folder workflow. They currently contain hardcoded paths for
another machine and `watchdog`/`streamlit` are not declared in
`requirements.txt`. They are not part of the verified setup. Before using
them, replace the hardcoded paths with environment-based paths, add their
dependencies, and ensure they use the same database and embedding backend as
the main CLI.

## API Examples

Ingest a folder:

```powershell
curl.exe -X POST http://localhost:8000/ingest `
  -H "Content-Type: application/json" `
  -d '{"folder":"data/documents"}'
```

Ask a question:

```powershell
curl.exe -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d '{"question":"What is our refund policy?"}'
```

Benchmark retrieval-only latency (embeds, searches, and ranks but never
calls the LLM, so no answer is generated):

```powershell
curl.exe -X POST http://localhost:8000/query/latency `
  -H "Content-Type: application/json" `
  -d '{"question":"What is our refund policy?","runs":5,"top_k":5}'
```

The response reports per-run times (`run_times_ms`), avg/min/max, a
per-stage breakdown (`stages`: query embedding, vector search, keyword
search, fusion, ranking), and the chunks/scores the last run retrieved.

Check status:

```powershell
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/stats
```

## Testing

The test suite uses deterministic fakes for embedding and LLM calls, and
creates temporary documents and databases. It is intended to run offline:

```powershell
python -m pytest -q
python -m pytest tests/test_ingestion.py -v
python -m pytest tests/test_api.py -v
python -m pytest tests/test_performance.py -v -s
```

The tests cover loaders, chunking, embeddings, vector storage, ingestion,
retrieval, the RAG pipeline, LLM behavior, API validation, integration, and
performance. Tests that use temporary SQLite stores do not require the live
PostgreSQL database or external model credentials.

## Troubleshooting

**`No module named pytest` or `No module named dotenv`**

The terminal is using a different interpreter from the virtual environment.
Activate `.venv` and verify:

```powershell
python -c "import sys; print(sys.executable)"
python -m pip install -r requirements.txt
```

**PostgreSQL connection failure**

Check `DATABASE_BACKEND`, `DATABASE_URL`, network access, credentials, and
that the PostgreSQL `vector` extension is installed.

**Google authentication failure**

Check `GCP_PROJECT_ID`, `GCP_LOCATION`, application-default credentials, and
that the Vertex AI API is enabled for the project.

**Embedding dimension mismatch**

The PostgreSQL schema expects 768 dimensions. Use a compatible model or create
a fresh database and update the schema and store configuration together.

## Security Notes

- Keep `.env`, service-account JSON files, and API keys out of version control.
- Rotate credentials if they have been pasted into chat, logs, or committed
  history.
- Keep `ALLOWED_INGEST_ROOT` restricted to the intended document directory.
- Configure explicit CORS origins for browser clients; leave CORS empty when it
  is not needed.
