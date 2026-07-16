"""Unit tests for src/retrieval (Phase 7)."""

from __future__ import annotations

import pytest

from src.embeddings.base_embedding import EmbeddingError
from src.embeddings.qwen_embedding import QwenEmbedding
from src.retrieval.retriever import Retriever, RetrievalError
from src.vectordb.vector_store import VectorStore, VectorStoreError


def _make_chunk(chunk_id: str, document_id: str, text: str, embedding: list[float], **meta) -> dict:
    metadata = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "id": chunk_id,
        "filename": f"{document_id}.txt",
        "page": None,
    }
    metadata.update(meta)
    return {"text": text, "embedding": embedding, "metadata": metadata}


@pytest.fixture
def populated_store(vector_store: VectorStore, embedder: QwenEmbedding) -> VectorStore:
    """A vector store pre-loaded with a few embedded chunks."""
    texts = [
        ("c1", "d1", "Refunds are available within 30 days of purchase."),
        ("c2", "d1", "Contact support to start a refund request."),
        ("c3", "d2", "The Pro plan costs $49 per month."),
    ]
    chunks = []
    for chunk_id, document_id, text in texts:
        embedding = embedder.embed(text)
        chunks.append(_make_chunk(chunk_id, document_id, text, embedding))
    vector_store.insert_many(chunks)
    return vector_store


class TestRetriever:
    def test_retrieve_with_scores_returns_ranked_chunks(
        self, populated_store: VectorStore, embedder: QwenEmbedding
    ) -> None:
        retriever = Retriever(
            embedder, populated_store, default_top_k=3, default_similarity_threshold=-1.0
        )

        results = retriever.retrieve_with_scores("refund policy")

        assert len(results) == 3
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        for result in results:
            assert "text" in result
            assert "metadata" in result

    def test_retrieve_and_search_are_aliases(
        self, populated_store: VectorStore, embedder: QwenEmbedding
    ) -> None:
        retriever = Retriever(
            embedder, populated_store, default_top_k=3, default_similarity_threshold=-1.0
        )

        a = retriever.retrieve("refund policy")
        b = retriever.search("refund policy")
        c = retriever.retrieve_with_scores("refund policy")

        assert [r["text"] for r in a] == [r["text"] for r in b] == [r["text"] for r in c]

    def test_top_k_is_respected(self, populated_store: VectorStore, embedder: QwenEmbedding) -> None:
        retriever = Retriever(
            embedder, populated_store, default_top_k=5, default_similarity_threshold=-1.0
        )
        results = retriever.retrieve_with_scores("refund policy", top_k=1)
        assert len(results) == 1

    def test_similarity_threshold_filters_low_scores(
        self, populated_store: VectorStore, embedder: QwenEmbedding
    ) -> None:
        retriever = Retriever(embedder, populated_store, default_top_k=5)
        # A threshold above 1.0 can never be met by cosine similarity.
        results = retriever.retrieve_with_scores("refund policy", similarity_threshold=1.5)
        assert results == []

    def test_filename_filter_restricts_results(
        self, populated_store: VectorStore, embedder: QwenEmbedding
    ) -> None:
        retriever = Retriever(
            embedder, populated_store, default_top_k=5, default_similarity_threshold=-1.0
        )
        results = retriever.retrieve_with_scores("plan pricing", filename="d2.txt")
        assert all(r["metadata"]["filename"] == "d2.txt" for r in results)

    def test_metadata_filter_restricts_results(
        self, populated_store: VectorStore, embedder: QwenEmbedding
    ) -> None:
        retriever = Retriever(
            embedder, populated_store, default_top_k=5, default_similarity_threshold=-1.0
        )
        results = retriever.retrieve_with_scores(
            "refund policy", metadata_filter={"document_id": "d1"}
        )
        assert all(r["metadata"]["document_id"] == "d1" for r in results)

    def test_empty_question_returns_empty_list(
        self, populated_store: VectorStore, embedder: QwenEmbedding
    ) -> None:
        retriever = Retriever(embedder, populated_store)
        assert retriever.retrieve_with_scores("   ") == []

    def test_empty_store_returns_empty_list(
        self, vector_store: VectorStore, embedder: QwenEmbedding
    ) -> None:
        retriever = Retriever(embedder, vector_store, default_similarity_threshold=-1.0)
        assert retriever.retrieve_with_scores("anything") == []

    def test_embedding_failure_raises_retrieval_error(
        self, populated_store: VectorStore, embedder: QwenEmbedding, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken_embed(text: str) -> list[float]:
            raise EmbeddingError("simulated outage")

        monkeypatch.setattr(embedder, "embed", broken_embed)
        retriever = Retriever(embedder, populated_store)

        with pytest.raises(RetrievalError):
            retriever.retrieve_with_scores("refund policy")

    def test_vector_store_failure_raises_retrieval_error(
        self, populated_store: VectorStore, embedder: QwenEmbedding, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken_search(embedding, top_k=None):  # noqa: ANN001
            raise VectorStoreError("simulated db error")

        monkeypatch.setattr(populated_store, "similarity_search", broken_search)
        retriever = Retriever(embedder, populated_store)

        with pytest.raises(RetrievalError):
            retriever.retrieve_with_scores("refund policy")
