"""Abstract interface for embedding providers.

Any future embedding backend (OpenAI, local sentence-transformers,
another vendor's API, etc.) can implement this interface without
touching the rest of the pipeline, as long as it returns the canonical
``EmbeddedChunk`` shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict

from src.chunking.chunker import Chunk


class EmbeddedChunk(TypedDict):
    """The canonical shape returned for every embedded chunk."""

    text: str
    embedding: list[float]
    metadata: dict[str, Any]


class EmbeddingError(Exception):
    """Raised when the embedding backend fails after exhausting retries."""


class BaseEmbedding(ABC):
    """Abstract base class all embedding providers must implement."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single piece of text and return its vector."""
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a batch of chunks and return them with vectors attached."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int | None:
        """Return the embedding vector dimensionality once known.

        Returns ``None`` until at least one embedding call has been
        made, since different models return different dimensions and
        we never hardcode this value.
        """
        raise NotImplementedError
