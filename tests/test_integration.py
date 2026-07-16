"""End-to-end integration tests (Phase 10).

Exercises the complete pipeline exactly as a real deployment would:

    Read PDFs -> Chunk -> Embed -> Store -> Query -> Retrieve -> Generate Answer

Generates 10 synthetic sample PDFs (via reportlab) covering distinct
topics, ingests all of them through the real ``IngestionService``, then
issues several queries through the real ``RAGPipeline`` and verifies
chunk counts, storage, retrieval quality, and answer generation.

Embedding/LLM APIs are patched to deterministic offline fakes (see
``tests/conftest.py``) so this suite runs fully offline and
reproducibly, while every other layer (loaders, chunker, vector store,
retriever, prompt builder) is exercised for real.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from src.embeddings.qwen_embedding import QwenEmbedding
from src.ingestion.ingestion_service import IngestionService
from src.llm.sllm import SLLMChat
from src.rag.rag_pipeline import RAGPipeline
from src.retrieval.retriever import Retriever
from src.vectordb.vector_store import VectorStore

# 10 distinct topics, each rendered as its own multi-page PDF.
_SAMPLE_PDF_CONTENT = [
    ("refund_policy.pdf", "Refunds are available within 30 days of purchase for unused subscriptions."),
    ("pricing.pdf", "The Pro plan costs $49 per month and includes unlimited document ingestion."),
    ("support_hours.pdf", "Support is available Monday through Friday, 9am to 6pm Eastern Time."),
    ("warranty.pdf", "Hardware warranty coverage extends for 12 months from the original purchase date."),
    ("onboarding.pdf", "New employees complete onboarding training during their first two weeks."),
    ("security.pdf", "All customer data is encrypted at rest using AES-256 and in transit using TLS 1.3."),
    ("api_limits.pdf", "The public API allows up to 1000 requests per minute per API key."),
    ("shipping.pdf", "Standard shipping takes 5 to 7 business days within the continental United States."),
    ("privacy.pdf", "User data is never sold to third parties and is retained for no more than 2 years."),
    ("billing_cycle.pdf", "Invoices are generated on the first day of each calendar month."),
]

# A simple bag-of-words vectorizer, built over the vocabulary of the
# sample content itself. Unlike the hash-based fake in conftest.py
# (deterministic but not semantically meaningful - fine for plumbing
# tests), this gives cosine similarity an actual topical signal so
# "retrieval quality" assertions in this integration suite are
# meaningful, while still running fully offline.
_VOCAB = sorted(
    {
        word
        for _, sentence in _SAMPLE_PDF_CONTENT
        for word in re.findall(r"[a-z0-9]+", sentence.lower())
    }
)


def _bag_of_words_vector(text: str) -> list[float]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    counts = [float(words.count(term)) for term in _VOCAB]
    norm = sum(c * c for c in counts) ** 0.5
    if norm == 0:
        return [0.0] * len(_VOCAB)
    return [c / norm for c in counts]


@pytest.fixture
def semantic_embedder(monkeypatch: pytest.MonkeyPatch) -> QwenEmbedding:
    """A QwenEmbedding wired to a bag-of-words fake instead of the
    purely-random fake in conftest.py, so retrieval-quality assertions
    in this suite reflect actual topical relevance.
    """

    def fake_call_api(self: QwenEmbedding, texts: list[str]) -> list[list[float]]:
        vectors = [_bag_of_words_vector(text) for text in texts]
        if vectors and self._dimensions is None:
            self._dimensions = len(vectors[0])
        return vectors

    monkeypatch.setattr(QwenEmbedding, "_call_api", fake_call_api)
    return QwenEmbedding(api_key="test-key", base_url="https://fake-sllm.test/v1", model="test-embed")


@pytest.fixture
def ten_sample_pdfs(tmp_path: Path) -> Path:
    """Generate 10 synthetic multi-page PDFs covering distinct topics."""
    pdf_dir = tmp_path / "sample_pdfs"
    pdf_dir.mkdir()

    for filename, sentence in _SAMPLE_PDF_CONTENT:
        path = pdf_dir / filename
        pdf = canvas.Canvas(str(path))
        pdf.drawString(72, 750, f"Document: {filename}")
        pdf.drawString(72, 730, sentence)
        pdf.showPage()
        pdf.drawString(72, 750, "Page 2")
        pdf.drawString(72, 730, "Additional supporting detail on the same topic follows here.")
        pdf.showPage()
        pdf.save()

    return pdf_dir


@pytest.fixture
def rag_pipeline(vector_store: VectorStore, semantic_embedder: QwenEmbedding, llm: SLLMChat) -> RAGPipeline:
    retriever = Retriever(
        embedder=semantic_embedder, vector_store=vector_store, default_top_k=5, default_similarity_threshold=-1.0
    )
    return RAGPipeline(retriever=retriever, llm=llm, top_k=5, similarity_threshold=-1.0)


class TestFullPipelineIntegration:
    def test_ingests_ten_pdfs_with_correct_chunk_and_vector_counts(
        self, ten_sample_pdfs: Path, vector_store: VectorStore, semantic_embedder: QwenEmbedding
    ) -> None:
        service = IngestionService(vector_store=vector_store, embedder=semantic_embedder)

        result = service.run(ten_sample_pdfs)

        assert result.documents_loaded == 10
        assert result.new_documents == 10
        assert result.chunks_created > 0
        assert result.embeddings_generated == result.chunks_created
        assert result.vectors_stored == result.chunks_created
        assert vector_store.count() == result.chunks_created

        # Every PDF should be represented in the registry with >0 chunks.
        registered = vector_store.list_ingested_documents()
        assert len(registered) == 10
        assert all(doc["chunk_count"] > 0 for doc in registered)

    def test_query_after_ingestion_retrieves_relevant_source(
        self, ten_sample_pdfs: Path, vector_store: VectorStore, semantic_embedder: QwenEmbedding, rag_pipeline: RAGPipeline
    ) -> None:
        IngestionService(vector_store=vector_store, embedder=semantic_embedder).run(ten_sample_pdfs)

        response = rag_pipeline.answer("What is the refund policy?")

        assert response.used_llm is True
        assert response.answer
        assert len(response.chunks_used) > 0
        filenames = {source.filename for source in response.sources}
        assert "refund_policy.pdf" in filenames

    def test_multiple_queries_each_retrieve_and_answer(
        self, ten_sample_pdfs: Path, vector_store: VectorStore, semantic_embedder: QwenEmbedding, rag_pipeline: RAGPipeline
    ) -> None:
        IngestionService(vector_store=vector_store, embedder=semantic_embedder).run(ten_sample_pdfs)

        questions = [
            "How much does the Pro plan cost?",
            "What are the support hours?",
            "How long is the hardware warranty?",
            "How is customer data encrypted?",
            "How many API requests are allowed per minute?",
        ]

        for question in questions:
            response = rag_pipeline.answer(question)
            assert response.used_llm is True
            assert response.answer
            assert len(response.chunks_used) > 0

    def test_reingesting_same_pdfs_is_idempotent(
        self, ten_sample_pdfs: Path, vector_store: VectorStore, semantic_embedder: QwenEmbedding
    ) -> None:
        service = IngestionService(vector_store=vector_store, embedder=semantic_embedder)
        first = service.run(ten_sample_pdfs)
        second = service.run(ten_sample_pdfs)

        assert second.new_documents == 0
        assert second.duplicates_skipped == 10
        assert vector_store.count() == first.chunks_created
