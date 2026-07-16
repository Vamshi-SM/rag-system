"""FastAPI dependency-injection helpers.

All expensive objects (embedder, LLM client, vector store, retriever,
RAG pipeline, ingestion service) are created exactly once, during the
app's lifespan startup (see ``src/api/main.py``), and stored on
``app.state``. These dependency functions just pull them back out for
each request - no per-request construction, no duplicated wiring logic.
"""

from __future__ import annotations

from fastapi import Request

from src.embeddings.base_embedding import BaseEmbedding
from src.ingestion.ingestion_service import IngestionService
from src.llm.base_llm import BaseLLM
from src.rag.rag_pipeline import RAGPipeline
from src.retrieval.retriever import Retriever
from src.vectordb.vector_store import VectorStore


def get_vector_store(request: Request) -> VectorStore:
    """Return the app-wide vector store instance."""
    return request.app.state.vector_store


def get_embedder(request: Request) -> BaseEmbedding:
    """Return the app-wide embedding provider instance."""
    return request.app.state.embedder


def get_llm(request: Request) -> BaseLLM:
    """Return the app-wide LLM provider instance."""
    return request.app.state.llm


def get_retriever(request: Request) -> Retriever:
    """Return the app-wide retriever instance."""
    return request.app.state.retriever


def get_rag_pipeline(request: Request) -> RAGPipeline:
    """Return the app-wide RAG pipeline instance."""
    return request.app.state.rag_pipeline


def get_ingestion_service(request: Request) -> IngestionService:
    """Return the app-wide ingestion service instance."""
    return request.app.state.ingestion_service
