"""FastAPI application entrypoint.

Run with:

    uvicorn src.api.main:app --reload

All expensive objects (embedding client, LLM client, vector store,
retriever, RAG pipeline, ingestion service) are constructed once during
the lifespan startup event and reused across requests via
``src/api/dependencies.py`` - never rebuilt per-request.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

# Allow running as `uvicorn src.api.main:app` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.exceptions import register_exception_handlers
from src.api.middleware import RequestLoggingMiddleware
from src.api.routes import router
from src.config import settings
from src.embeddings.qwen_embedding import QwenEmbedding
from src.ingestion.ingestion_service import IngestionService
from src.llm.sllm import SLLMChat
from src.rag.rag_pipeline import RAGPipeline
from src.retrieval.retriever import Retriever
from src.utils.logger import get_logger
from src.vectordb.vector_store import VectorStore

logger = get_logger(__name__)

TAGS_METADATA = [
    {"name": "Ingestion", "description": "Load, chunk, embed, and store documents."},
    {"name": "Query", "description": "Ask questions answered from ingested documents (RAG)."},
    {"name": "Monitoring", "description": "Health and usage statistics."},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build all shared, long-lived components once at startup and
    tear them down cleanly at shutdown.
    """
    settings.ensure_directories()
    logger.info("Starting RAG API - initializing shared components...")

    embedder = QwenEmbedding(
        api_key=settings.sllm_api_key,
        base_url=settings.sllm_base_url,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        timeout=settings.embedding_timeout,
        max_retries=settings.embedding_max_retries,
    )
    vector_store = VectorStore(database_path=settings.database_path, default_top_k=settings.top_k)
    llm = SLLMChat(
        api_key=settings.sllm_api_key,
        base_url=settings.sllm_base_url,
        model=settings.chat_model,
        timeout=settings.request_timeout,
        max_retries=settings.llm_max_retries,
    )
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        default_top_k=settings.top_k,
        default_similarity_threshold=settings.similarity_threshold,
        candidate_multiplier=settings.retrieval_candidate_multiplier,
    )
    rag_pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        top_k=settings.top_k,
        similarity_threshold=settings.similarity_threshold,
    )
    ingestion_service = IngestionService(vector_store=vector_store, embedder=embedder)

    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.llm = llm
    app.state.retriever = retriever
    app.state.rag_pipeline = rag_pipeline
    app.state.ingestion_service = ingestion_service

    logger.info("RAG API ready.")
    try:
        yield
    finally:
        logger.info("Shutting down RAG API...")
        vector_store.close()


def create_app() -> FastAPI:
    """Application factory, so tests can build isolated app instances."""
    app = FastAPI(
        title="RAG System API",
        description=(
            "Production REST API for the RAG pipeline: document ingestion, "
            "retrieval-augmented question answering, health, and usage stats."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
    )

    app.add_middleware(RequestLoggingMiddleware)

    if settings.allowed_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    register_exception_handlers(app)
    app.include_router(router)

    return app


app = create_app()
