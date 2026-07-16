"""Phase 7 - Retrieval module.

``Retriever`` wraps the existing (Phase 2-5) embedding provider and
vector store behind a small, question-oriented API. It doesn't
reimplement anything - it composes ``BaseEmbedding`` and
``VectorStore`` as-is.
"""

from __future__ import annotations

from typing import Any, TypedDict

from src.embeddings.base_embedding import BaseEmbedding, EmbeddingError
from src.utils.logger import get_logger
from src.vectordb.vector_store import VectorStore, VectorStoreError

logger = get_logger(__name__)


class RetrievedChunk(TypedDict):
    """A single retrieved chunk, ready to hand to a prompt builder."""

    text: str
    score: float
    metadata: dict[str, Any]


class RetrievalError(Exception):
    """Raised when retrieval fails due to an upstream error (embedding
    API down, vector store error, etc.) as opposed to simply finding no
    relevant results, which is a normal, non-error outcome.
    """


class Retriever:
    """Retrieves the most relevant chunks for a natural-language question.

    Flow: question -> embed -> vector similarity search -> filter/sort
    -> ranked chunks.

    Args:
        embedder: Any embedding provider implementing ``BaseEmbedding``
            (reuses the Phase 4 embedding module).
        vector_store: The Phase 5 ``VectorStore`` to search against.
        default_top_k: Default number of chunks returned.
        default_similarity_threshold: Minimum similarity score (0-1) a
            chunk must have to be included in results. Chunks below
            this are dropped rather than passed to the LLM, which is
            what keeps low-relevance context out of the prompt.
        candidate_multiplier: How many extra candidates to pull from
            the vector store (``top_k * candidate_multiplier``) before
            applying metadata/filename filters, so filtering doesn't
            starve the final result count.
    """

    def __init__(
        self,
        embedder: BaseEmbedding,
        vector_store: VectorStore,
        default_top_k: int = 5,
        default_similarity_threshold: float = 0.7,
        candidate_multiplier: int = 4,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.default_similarity_threshold = default_similarity_threshold
        self.candidate_multiplier = max(1, candidate_multiplier)

    def retrieve_with_scores(
        self,
        question: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
        filename: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top-K most relevant chunks, with similarity scores.

        Args:
            question: The natural-language query.
            top_k: Number of chunks to return. Defaults to
                ``self.default_top_k``.
            similarity_threshold: Minimum similarity (0-1) to keep a
                chunk. Defaults to ``self.default_similarity_threshold``.
            metadata_filter: Optional ``{key: value}`` pairs that must
                all match a chunk's metadata (exact match). Applied
                after the vector search, as a straightforward hook for
                future hybrid search / re-ranking.
            filename: Optional exact filename to restrict results to.

        Returns:
            Chunks sorted by descending similarity score, each capped
            at ``top_k`` and above ``similarity_threshold``. Returns an
            empty list (not an error) if nothing meets the criteria.

        Raises:
            RetrievalError: If the embedding call or vector search
                itself fails (as opposed to simply finding no results).
        """
        if not question or not question.strip():
            logger.warning("retrieve_with_scores called with an empty question")
            return []

        k = top_k or self.default_top_k
        threshold = (
            self.default_similarity_threshold
            if similarity_threshold is None
            else similarity_threshold
        )
        candidate_k = k * self.candidate_multiplier if (metadata_filter or filename) else k

        try:
            query_embedding = self.embedder.embed(question)
        except EmbeddingError as exc:
            logger.error("Retrieval failed: could not embed question: %s", exc)
            raise RetrievalError(f"Could not embed question: {exc}") from exc

        try:
            raw_results = self.vector_store.similarity_search(query_embedding, top_k=candidate_k)
        except VectorStoreError as exc:
            logger.error("Retrieval failed: vector store search error: %s", exc)
            raise RetrievalError(f"Vector store search failed: {exc}") from exc

        filtered: list[RetrievedChunk] = []
        for result in raw_results:
            if result["similarity"] < threshold:
                continue
            metadata = result["metadata"]

            if filename and metadata.get("filename") != filename:
                continue
            if metadata_filter and not all(
                metadata.get(key) == value for key, value in metadata_filter.items()
            ):
                continue

            filtered.append(
                {"text": result["text"], "score": result["similarity"], "metadata": metadata}
            )

        filtered.sort(key=lambda chunk: chunk["score"], reverse=True)
        top_results = filtered[:k]

        logger.info(
            "Retrieved %d/%d chunks for question (top_k=%d, threshold=%.2f)",
            len(top_results),
            len(raw_results),
            k,
            threshold,
        )
        return top_results

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
        filename: str | None = None,
    ) -> list[RetrievedChunk]:
        """Alias for :meth:`retrieve_with_scores`.

        Kept as a separate, explicitly-named method (rather than just
        documenting ``retrieve_with_scores``) because callers building
        prompts often read more clearly as ``retriever.retrieve(...)``,
        while callers building debug/inspection UIs read more clearly
        as ``retriever.retrieve_with_scores(...)``. Both return the same
        shape - scores are always included since dropping them would
        lose information for no benefit.
        """
        return self.retrieve_with_scores(
            question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            metadata_filter=metadata_filter,
            filename=filename,
        )

    def search(
        self,
        question: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
        filename: str | None = None,
    ) -> list[RetrievedChunk]:
        """Convenience alias for :meth:`retrieve_with_scores`.

        Provided so callers can use whichever verb reads best
        (``search`` / ``retrieve`` / ``retrieve_with_scores``) - all
        three are equivalent.
        """
        return self.retrieve_with_scores(
            question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            metadata_filter=metadata_filter,
            filename=filename,
        )
