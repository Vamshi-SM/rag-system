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
# /retrieve
# --------------------------------------------------------------------- #


class RetrieveRequest(BaseModel):
    """Query parameters for ``GET /retrieve`` (retrieval only, no LLM)."""

    question: str = Field(..., min_length=1, description="The natural-language query to retrieve chunks for.")
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
            "examples": [{"question": "What is the refund policy?", "top_k": 3}]
        }
    }


class RetrievedChunkModel(BaseModel):
    """A single retrieved chunk with its source attribution."""

    chunk_id: str | None = Field(None, description="Stable chunk identifier, when available.")
    text: str = Field(..., description="The chunk text as stored.")
    filename: str = Field(..., description="Source document filename.")
    page: int | None = Field(None, description="Page number for paginated sources (PDF).")
    score: float = Field(..., description="Fused relevance score of the chunk.")


class RetrieveResponse(BaseModel):
    """Response body for ``POST /retrieve``."""

    query: str = Field(..., description="The query that was embedded and searched.")
    chunks: list[RetrievedChunkModel] = Field(default_factory=list)
    retrieved_chunks: int = Field(..., description="Number of chunks returned.")
    latency: str = Field(..., description="Wall-clock time for the retrieval step only.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is the refund policy?",
                    "chunks": [
                        {
                            "chunk_id": "c0a8...",
                            "text": "Refunds are available within 30 days of purchase...",
                            "filename": "sample_refund_policy.txt",
                            "page": None,
                            "score": 0.94,
                        }
                    ],
                    "retrieved_chunks": 1,
                    "latency": "0.31 sec",
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
