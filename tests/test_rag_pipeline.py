"""Unit tests for src/rag (Phase 8): prompt, response, and pipeline."""

from __future__ import annotations

import pytest

from src.embeddings.base_embedding import EmbeddingError
from src.embeddings.qwen_embedding import QwenEmbedding
from src.llm.base_llm import LLMError
from src.llm.sllm import SLLMChat
from src.rag.prompt import build_prompt, format_context
from src.rag.rag_pipeline import RAGPipeline
from src.rag.response import NO_CONTEXT_ANSWER, build_sources
from src.retrieval.retriever import Retriever
from src.vectordb.vector_store import VectorStore


def _make_chunk(chunk_id: str, document_id: str, text: str, embedding: list[float], page=None) -> dict:
    return {
        "text": text,
        "embedding": embedding,
        "metadata": {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "id": chunk_id,
            "filename": f"{document_id}.txt",
            "page": page,
        },
    }


class TestPromptBuilder:
    def test_format_context_labels_filename_and_page(self) -> None:
        chunks = [
            {"text": "Refunds within 30 days.", "score": 0.9, "metadata": {"filename": "a.pdf", "page": 2}}
        ]
        context = format_context(chunks)
        assert "a.pdf, page 2" in context
        assert "Refunds within 30 days." in context

    def test_format_context_handles_missing_page(self) -> None:
        chunks = [{"text": "Some text.", "score": 0.9, "metadata": {"filename": "notes.txt"}}]
        context = format_context(chunks)
        assert "notes.txt" in context
        assert "page" not in context.split("notes.txt")[1].split("\n")[0]

    def test_format_context_empty_chunks(self) -> None:
        assert "no relevant context" in format_context([]).lower()

    def test_build_prompt_includes_instructions_context_and_question(self) -> None:
        chunks = [{"text": "context text", "score": 0.9, "metadata": {"filename": "a.txt"}}]
        prompt = build_prompt("What is the policy?", chunks)

        assert "Answer ONLY using the provided context" in prompt
        assert "context text" in prompt
        assert "What is the policy?" in prompt
        assert "Context" in prompt
        assert "Question" in prompt


class TestBuildSources:
    def test_deduplicates_same_filename_and_page(self) -> None:
        chunks = [
            {"text": "a", "score": 0.9, "metadata": {"filename": "doc.pdf", "page": 1}},
            {"text": "b", "score": 0.8, "metadata": {"filename": "doc.pdf", "page": 1}},
            {"text": "c", "score": 0.7, "metadata": {"filename": "doc.pdf", "page": 2}},
        ]
        sources = build_sources(chunks)
        assert len(sources) == 2

    def test_source_str_format(self) -> None:
        chunks = [{"text": "a", "score": 0.9, "metadata": {"filename": "doc.pdf", "page": 3}}]
        sources = build_sources(chunks)
        assert str(sources[0]) == "doc.pdf (page 3)"


@pytest.fixture
def populated_store(vector_store: VectorStore, embedder: QwenEmbedding) -> VectorStore:
    text = "Refunds are available within 30 days of purchase for unused subscription time."
    embedding = embedder.embed(text)
    vector_store.insert_many([_make_chunk("c1", "d1", text, embedding, page=1)])
    return vector_store


@pytest.fixture
def pipeline(populated_store: VectorStore, embedder: QwenEmbedding, llm: SLLMChat) -> RAGPipeline:
    retriever = Retriever(embedder, populated_store, default_top_k=3, default_similarity_threshold=-1.0)
    return RAGPipeline(retriever=retriever, llm=llm, top_k=3, similarity_threshold=-1.0)


class TestRAGPipeline:
    def test_answer_returns_grounded_response_with_sources(self, pipeline: RAGPipeline) -> None:
        response = pipeline.answer("What is the refund policy?")

        assert response.used_llm is True
        assert response.answer
        assert response.error is None
        assert len(response.sources) == 1
        assert response.sources[0].filename == "d1.txt"
        assert response.sources[0].page == 1
        assert len(response.chunks_used) == 1

    def test_empty_question_short_circuits(self, pipeline: RAGPipeline) -> None:
        response = pipeline.answer("   ")
        assert response.used_llm is False
        assert response.error == "empty_question"

    def test_no_matching_chunks_short_circuits_without_calling_llm(
        self, embedder: QwenEmbedding, vector_store: VectorStore, llm: SLLMChat
    ) -> None:
        retriever = Retriever(embedder, vector_store, default_similarity_threshold=-1.0)
        pipeline = RAGPipeline(retriever=retriever, llm=llm)

        response = pipeline.answer("Anything at all")

        assert response.used_llm is False
        assert response.answer == NO_CONTEXT_ANSWER
        assert response.sources == []

    def test_embedding_failure_yields_safe_fallback(
        self, pipeline: RAGPipeline, embedder: QwenEmbedding, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken_embed(text: str) -> list[float]:
            raise EmbeddingError("simulated outage")

        monkeypatch.setattr(embedder, "embed", broken_embed)

        response = pipeline.answer("What is the refund policy?")

        assert response.used_llm is False
        assert response.error is not None
        assert "retrieval_error" in response.error
        assert response.answer  # never empty/None

    def test_llm_failure_yields_safe_fallback_but_keeps_sources(
        self, pipeline: RAGPipeline, llm: SLLMChat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken_chat(messages):  # noqa: ANN001
            raise LLMError("simulated LLM outage")

        monkeypatch.setattr(llm, "generate", broken_chat)

        response = pipeline.answer("What is the refund policy?")

        assert response.used_llm is False
        assert response.error is not None
        assert "llm_error" in response.error
        assert len(response.sources) == 1  # retrieval still succeeded

    def test_never_raises_even_on_unexpected_exception_types(
        self, pipeline: RAGPipeline, embedder: QwenEmbedding, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken_embed(text: str) -> list[float]:
            raise RuntimeError("totally unexpected failure")

        monkeypatch.setattr(embedder, "embed", broken_embed)

        # Even an unexpected (non-EmbeddingError) exception from a
        # dependency must not propagate out of the pipeline - "never
        # crash the pipeline" applies to any failure mode, not just the
        # ones we anticipated.
        response = pipeline.answer("What is the refund policy?")

        assert response.used_llm is False
        assert response.error is not None
        assert "unexpected_retrieval_error" in response.error
        assert response.answer
