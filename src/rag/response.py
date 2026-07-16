"""Phase 8 - Response modeling for the RAG pipeline.

``RAGResponse`` is the single object every RAG query returns,
regardless of whether it went all the way to the LLM or short-circuited
(e.g. because retrieval found nothing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.retrieval.retriever import RetrievedChunk

NO_CONTEXT_ANSWER = (
    "The information is not available in the documents I have access to."
)


@dataclass
class Source:
    """A single source citation for display to the user."""

    filename: str
    page: int | None = None

    def __str__(self) -> str:
        return f"{self.filename} (page {self.page})" if self.page is not None else self.filename


@dataclass
class RAGResponse:
    """The result of a single RAG query.

    Attributes:
        answer: The final natural-language answer.
        sources: De-duplicated list of sources the answer was (or
            would have been) grounded in.
        chunks_used: The raw retrieved chunks, for callers that want
            more detail than just filename/page (e.g. showing scores).
        used_llm: Whether the LLM was actually called. ``False`` when
            the pipeline short-circuited due to empty retrieval.
        elapsed_seconds: Total wall-clock time for the query.
        error: Populated with a human-readable message if something
            went wrong; ``answer`` will still contain a safe fallback
            message in that case so callers never have to handle
            ``None``.
    """

    answer: str
    sources: list[Source] = field(default_factory=list)
    chunks_used: list[RetrievedChunk] = field(default_factory=list)
    used_llm: bool = True
    elapsed_seconds: float = 0.0
    error: str | None = None

    def format_sources(self) -> str:
        """Render sources as a human-readable block, e.g. for CLI display."""
        if not self.sources:
            return "(no sources)"
        return "\n".join(f"- {source}" for source in self.sources)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (e.g. for an API JSON response)."""
        return {
            "answer": self.answer,
            "sources": [{"filename": s.filename, "page": s.page} for s in self.sources],
            "chunks_used": self.chunks_used,
            "used_llm": self.used_llm,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }


def build_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    """Build a de-duplicated, order-preserving list of sources from chunks."""
    seen: set[tuple[str, int | None]] = set()
    sources: list[Source] = []

    for chunk in chunks:
        filename = chunk["metadata"].get("filename", "unknown source")
        page = chunk["metadata"].get("page")
        key = (filename, page)
        if key not in seen:
            seen.add(key)
            sources.append(Source(filename=filename, page=page))

    return sources
