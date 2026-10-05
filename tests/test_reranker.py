"""Unit tests for the cross-encoder reranker integration (Phase 7b)."""

from __future__ import annotations

import pytest

from src.retrieval.retriever import Retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.vectordb.vector_store import VectorStore


class FakeReranker:
    """Deterministic stand-in: scores = length of the longest word in text."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def rerank(self, question, results, top_k):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        candidates = results[:20]
        for r in candidates:
            r["score"] = float(len(max(r["text"].split(), key=len)))
            r["rerank_score"] = r["score"]
        candidates.sort(key=lambda r: r["score"], reverse=True)
        return candidates[:top_k]


@pytest.fixture
def populated_store(vector_store: VectorStore, embedder) -> VectorStore:
    texts = [
        ("c1", "d1", "Refunds are available within thirty days of purchase."),
        ("c2", "d1", "Short note."),
        ("c3", "d2", "The Pro plan costs forty nine dollars per month."),
        ("c4", "d2", "Another chunk mentioning refunds and the support team."),
    ]
    chunks = []
    for chunk_id, document_id, text in texts:
        embedding = embedder.embed(text)
        chunks.append({
            "text": text,
            "embedding": embedding,
            "metadata": {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "id": chunk_id,
                "filename": f"{document_id}.txt",
            },
        })
    vector_store.insert_many(chunks)
    return vector_store


class TestRerankerIntegration:
    def test_reranker_changes_order_and_scores(
        self, populated_store: VectorStore, embedder
    ) -> None:
        fake = FakeReranker()
        retriever = Retriever(
            embedder, populated_store, default_top_k=3,
            default_similarity_threshold=-1.0, reranker=fake,
        )
        results = retriever.retrieve_with_scores("refund policy")
        assert fake.calls == 1
        assert len(results) == 3
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert all("rerank_score" in r for r in results)

    def test_reranker_failure_falls_back_to_fusion_order(
        self, populated_store: VectorStore, embedder
    ) -> None:
        failing = FakeReranker(fail=True)
        retriever = Retriever(
            embedder, populated_store, default_top_k=3,
            default_similarity_threshold=-1.0, reranker=failing,
        )
        results = retriever.retrieve_with_scores("refund policy")
        assert len(results) == 3  # graceful degradation, not an exception

    def test_no_reranker_keeps_fusion_scores(
        self, populated_store: VectorStore, embedder
    ) -> None:
        retriever = Retriever(
            embedder, populated_store, default_top_k=3,
            default_similarity_threshold=-1.0,
        )
        results = retriever.retrieve_with_scores("refund policy")
        assert all("rerank_score" not in r for r in results)


class TestCrossEncoderReranker:
    def test_rerank_empty_results(self) -> None:
        r = CrossEncoderReranker()
        assert r.rerank("q", [], top_k=5) == []

    def test_rerank_returns_top_k(self) -> None:
        r = CrossEncoderReranker()
        results = [
            {"text": f"chunk {i}", "score": 0.0, "metadata": {}}
            for i in range(10)
        ]
        out = r.rerank("which chunk is relevant?", results, top_k=3)
        assert len(out) == 3
        scores = [x["score"] for x in out]
        assert scores == sorted(scores, reverse=True)