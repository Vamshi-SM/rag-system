"""Embedding provider factory.

Single source of truth for constructing the embedding backend selected by
``EMBEDDING_BACKEND`` in settings: "ollama", "local" (fastembed),
"google" (Vertex AI), or the default SharedLLM/Qwen-compatible provider.

Used by both the API lifespan (``src/api/main.py``) and the ingestion CLI
(``scripts/ingest.py``) so the two can never drift apart on which backend
gets wired up.
"""

from __future__ import annotations

from src.config import Settings
from src.embeddings.base_embedding import BaseEmbedding
from src.embeddings.google_embedding import GoogleEmbedding
from src.embeddings.local_embedding import LocalEmbedding
from src.embeddings.ollama_embedding import OllamaEmbedding
from src.embeddings.qwen_embedding import QwenEmbedding


def build_embedder(settings: Settings) -> BaseEmbedding:
    """Construct the embedding provider selected by ``EMBEDDING_BACKEND``."""
    if settings.embedding_backend == "ollama":
        return OllamaEmbedding(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
            batch_size=settings.embedding_batch_size,
            timeout=settings.embedding_timeout,
            max_retries=settings.embedding_max_retries,
        )
    if settings.embedding_backend == "local":
        return LocalEmbedding(
            model=settings.local_embedding_model,
            batch_size=settings.embedding_batch_size,
        )
    if settings.embedding_backend == "google":
        return GoogleEmbedding(
            project_id=settings.gcp_project_id,
            location=settings.gcp_location,
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            max_chars=settings.embedding_max_chars,
            max_request_tokens=settings.embedding_token_budget,
            concurrency=settings.embedding_concurrency,
        )
    return QwenEmbedding(
        api_key=settings.sllm_api_key,
        base_url=settings.sllm_base_url,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        timeout=settings.embedding_timeout,
        max_retries=settings.embedding_max_retries,
    )