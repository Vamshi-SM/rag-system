"""Pydantic v2 request/response schemas for the FastAPI app.

Every model includes a ``json_schema_extra`` example so they render
nicely in the auto-generated Swagger UI (``/docs``) and ReDoc
(``/redoc``).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------- #
# /ingest
# --------------------------------------------------------------------- #


class IngestRequest(BaseModel):
    """Request body for ``POST /ingest``."""

    folder: str = Field(
        ...,
        min_length=1,
        description="Path to a directory to recursively scan for supported documents.",
        examples=["./data/documents"],
    )

    @field_validator("folder")
    @classmethod
    def folder_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("folder must not be blank")
        return stripped

    model_config = {
        "json_schema_extra": {"examples": [{"folder": "./data/documents"}]}
    }


class IngestResponse(BaseModel):
    """Response body for ``POST /ingest``."""

    status: str = Field(..., description="'success' or 'partial' if some documents failed.")
    documents: int = Field(..., description="Number of new documents ingested (duplicates excluded).")
    chunks: int = Field(..., description="Number of chunks created from the new documents.")
    stored_vectors: int = Field(..., description="Number of vectors written to the store.")
    duplicates_skipped: int = Field(0, description="Number of documents skipped as already ingested.")
    processing_time: str = Field(..., description="Wall-clock time for the ingestion run.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "documents": 12,
                    "chunks": 415,
                    "stored_vectors": 415,
                    "duplicates_skipped": 0,
                    "processing_time": "4.8 sec",
                }
            ]
        }
    }


# --------------------------------------------------------------------- #
# /query
# --------------------------------------------------------------------- #


class QueryRequest(BaseModel):
    """Request body for ``POST /query``."""

    question: str = Field(..., min_length=1, description="The natural-language question to ask.")
    top_k: int | None = Field(None, ge=1, le=50, description="Override the default top-K chunks.")
    similarity_threshold: float | None = Field(
        None, ge=-1.0, le=1.0, description="Override the default minimum similarity score."
    )
    filename: str | None = Field(None, description="Restrict retrieval to a specific filename.")

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "How does Council GPT perform embeddings?"}]
        }
    }


class SourceModel(BaseModel):
    """A single source citation in a query response."""

    filename: str
    page: int | None = None
    score: float


class QueryResponse(BaseModel):
    """Response body for ``POST /query``."""

    answer: str
    sources: list[SourceModel] = Field(default_factory=list)
    retrieved_chunks: int
    latency: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "Council GPT performs embeddings using the Qwen embedding model...",
                    "sources": [
                        {"filename": "doc1.pdf", "page": 2, "score": 0.94},
                        {"filename": "doc3.pdf", "page": 8, "score": 0.89},
                    ],
                    "retrieved_chunks": 5,
                    "latency": "1.42 sec",
                }
            ]
        }
    }


# --------------------------------------------------------------------- #
# /query/latency
# --------------------------------------------------------------------- #


class LatencyRequest(BaseModel):
    """Request body for ``POST /query/latency`` (retrieval-only benchmark)."""

    question: str = Field(..., min_length=1, description="The natural-language question to ask.")
    top_k: int | None = Field(None, ge=1, le=50, description="Override the default top-K chunks.")
    similarity_threshold: float | None = Field(
        None, ge=-1.0, le=1.0, description="Override the default minimum similarity score."
    )
    filename: str | None = Field(None, description="Restrict retrieval to a specific filename.")
    runs: int = Field(
        1,
        ge=1,
        le=25,
        description="Number of retrieval passes to time (avg/min/max reported).",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "How does Council GPT perform embeddings?",
                    "top_k": 5,
                    "runs": 5,
                }
            ]
        }
    }


class StageTiming(BaseModel):
    """Per-stage timing, averaged over the measured runs."""

    stage: str = Field(..., description="Stage name (embed, vector search, fusion, rank).")
    milliseconds: float = Field(..., description="Average wall-clock time per run, in ms.")
    calls: int = Field(..., description="How many times the stage ran across all runs.")


class ChunkModel(BaseModel):
    """A retrieved chunk, with its text, for inspection of what the
    retriever actually returned."""

    filename: str
    page: int | None = None
    score: float
    text: str


class LatencyResponse(BaseModel):
    """Response body for ``POST /query/latency``.

    Retrieval-only: the LLM is never called, so the timings isolate the
    embed + search + fusion + ranking cost of a query.
    """

    runs: int = Field(..., description="Number of retrieval passes performed.")
    retrieved_chunks: int = Field(..., description="Chunks returned by the final run.")
    top_score: float | None = Field(None, description="Similarity score of the top chunk.")
    sources: list[SourceModel] = Field(default_factory=list)
    chunks: list[ChunkModel] = Field(
        default_factory=list,
        description="The chunks retrieved by the final run, including their full text.",
    )
    run_times_ms: list[float] = Field(..., description="Total retrieval time per run, in ms.")
    avg_ms: float
    min_ms: float
    max_ms: float
    stages: list[StageTiming] = Field(
        default_factory=list, description="Per-stage timings averaged per run."
    )
    latency: str = Field(..., description="Formatted average retrieval latency.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "runs": 5,
                    "retrieved_chunks": 5,
                    "top_score": 0.87,
                    "sources": [{"filename": "doc1.pdf", "page": 2, "score": 0.87}],
                    "chunks": [
                        {
                            "filename": "doc1.pdf",
                            "page": 2,
                            "score": 0.87,
                            "text": "Refunds are available within 30 days...",
                        }
                    ],
                    "run_times_ms": [142.1, 138.4, 141.9, 140.2, 139.8],
                    "avg_ms": 140.48,
                    "min_ms": 138.4,
                    "max_ms": 142.1,
                    "stages": [
                        {"stage": "2. query embedding", "milliseconds": 95.2, "calls": 5},
                        {"stage": "3. vector similarity search", "milliseconds": 12.6, "calls": 5},
                    ],
                    "latency": "0.14 sec",
                }
            ]
        }
    }


# --------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str = Field(..., description="'healthy' or 'degraded'.")
    database: str = Field(..., description="'connected' or 'disconnected'.")
    embedding_service: str = Field(..., description="'online' or 'offline'.")
    llm: str = Field(..., description="'online' or 'offline'.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "database": "connected",
                    "embedding_service": "online",
                    "llm": "online",
                }
            ]
        }
    }


# --------------------------------------------------------------------- #
# /stats
# --------------------------------------------------------------------- #


class StatsResponse(BaseModel):
    """Response body for ``GET /stats``."""

    documents: int
    chunks: int
    embeddings: int
    database_size: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"documents": 24, "chunks": 925, "embeddings": 925, "database_size": "18.0 MB"}
            ]
        }
    }


# --------------------------------------------------------------------- #
# Shared error shape
# --------------------------------------------------------------------- #


class ErrorResponse(BaseModel):
    """Structured error shape returned by every exception handler."""

    status: str = "error"
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"status": "error", "message": "Folder not found."}]
        }
    }
