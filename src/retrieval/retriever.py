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
from src.vectordb.vector_store import SearchResult, VectorStore, VectorStoreError

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


# Reciprocal Rank Fusion constant. A chunk at rank ``r`` (0-indexed) in a
# given result list contributes ``1 / (RRF_K + r)`` to its fused score. 60 is
# the value from the original RRF paper and works well in practice.
RRF_K = 60


def _fuse_rrf(
    vector_results: list[SearchResult], keyword_results: list[SearchResult]
) -> list[SearchResult]:
    """Fuse two ranked result lists via Reciprocal Rank Fusion (RRF).

    RRF is rank-based (not score-based), so it gracefully combines two
    systems whose scores are on incomparable scales (cosine similarity in
    [0, 1] vs. FTS5 BM25). A chunk appearing in both lists gets a higher
    fused score than one in only one list, and rank position within each
    list still matters.

    The fused RRF score is used for *ordering* and is returned in the
    ``similarity`` field (so the final ranked list's scores are monotonic
    with its order). The original dense cosine similarity is preserved in a
    ``cosine_similarity`` field so the retriever can apply a cosine-based
    similarity threshold on its original [0, 1] scale — RRF scores are tiny
    (~1/60) and would make a cosine threshold reject everything. Chunks that
    only appeared in the keyword list carry their normalized BM25 similarity
    as both ``similarity`` and ``cosine_similarity``.

    Returns results sorted by descending fused RRF score.
    """
    fused: dict[str, dict[str, Any]] = {}

    for rank, result in enumerate(vector_results):
        key = result["id"]
        rrf_score = 1.0 / (RRF_K + rank)
        if key not in fused:
            fused[key] = {"result": result, "rrf": 0.0, "cosine": result["similarity"]}
        fused[key]["rrf"] += rrf_score

    for rank, result in enumerate(keyword_results):
        key = result["id"]
        rrf_score = 1.0 / (RRF_K + rank)
        if key not in fused:
            # Only-keyword hit: use its normalized BM25 similarity as the
            # cosine proxy for thresholding.
            fused[key] = {"result": result, "rrf": 0.0, "cosine": result["similarity"]}
        fused[key]["rrf"] += rrf_score

    ordered = sorted(fused.values(), key=lambda item: item["rrf"], reverse=True)
    fused_results: list[SearchResult] = []
    for item in ordered:
        result = dict(item["result"])
        result["cosine_similarity"] = item["cosine"]
        result["similarity"] = item["rrf"]  # fused score → drives ordering
        result["distance"] = 1.0 - item["rrf"]
        fused_results.append(result)
    return fused_results


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

        k = top_k if top_k is not None else self.default_top_k
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

        # --- Dense (vector) search ---
        # Run the vector search first and let its errors propagate as
        # RetrievalError (preserving the contract relied on by callers/tests)
        # before attempting the keyword search below.
        try:
            vector_results = self.vector_store.similarity_search(
                query_embedding, top_k=candidate_k
            )
        except VectorStoreError as exc:
            logger.error("Retrieval failed: vector store search error: %s", exc)
            raise RetrievalError(f"Vector store search failed: {exc}") from exc

        # --- Sparse (keyword) search, fused via Reciprocal Rank Fusion ---
        # Keyword search is best-effort: if the FTS5 index is unavailable or
        # the query fails, we degrade to pure-vector results rather than
        # failing the whole retrieval.
        keyword_results: list[SearchResult] = []
        keyword_search = getattr(self.vector_store, "keyword_search", None)
        if keyword_search is not None:
            try:
                keyword_results = keyword_search(question, limit=candidate_k)
            except Exception as exc:  # noqa: BLE001 - keyword search must never break retrieval
                logger.warning("Keyword search failed, using vector-only results: %s", exc)
                keyword_results = []

        fused_results = _fuse_rrf(vector_results, keyword_results)

        # --- Threshold + metadata/filename filtering ---
        # Threshold against the dense cosine similarity (the meaningful
        # relevance signal on [0, 1]); rank by the fused RRF score so the
        # returned `score` is monotonic with the final order.
        filtered: list[RetrievedChunk] = []
        for result in fused_results:
            cosine = result.get("cosine_similarity", result["similarity"])
            if cosine < threshold:
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

        # RRF already returns a rank order, but filtering can drop entries,
        # so re-sort the surviving set by fused score for a stable top-k.
        filtered.sort(key=lambda chunk: chunk["score"], reverse=True)
        top_results = filtered[:k]

        logger.info(
            "Retrieved %d chunks for question "
            "(vector=%d, keyword=%d, fused=%d, top_k=%d, threshold=%.2f)",
            len(top_results),
            len(vector_results),
            len(keyword_results),
            len(fused_results),
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
