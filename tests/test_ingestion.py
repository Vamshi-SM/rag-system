"""Unit tests for src/ingestion (Phase 6, shared by CLI and API)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.embeddings.qwen_embedding import QwenEmbedding
from src.ingestion.ingestion_service import IngestionService
from src.vectordb.vector_store import VectorStore


class TestIngestionService:
    def test_ingests_all_supported_documents(
        self, sample_documents_dir: Path, vector_store: VectorStore, embedder: QwenEmbedding
    ) -> None:
        service = IngestionService(vector_store=vector_store, embedder=embedder)

        result = service.run(sample_documents_dir)

        # policy.txt, pricing.md, handbook.docx, manual.pdf = 4 supported;
        # notes.xyz is skipped as unsupported.
        assert result.documents_loaded == 4
        assert result.new_documents == 4
        assert result.duplicates_skipped == 0
        assert result.chunks_created > 0
        assert result.embeddings_generated == result.chunks_created
        assert result.vectors_stored == result.chunks_created
        assert result.status == "success"
        assert vector_store.count() == result.chunks_created

    def test_second_run_skips_all_as_duplicates(
        self, sample_documents_dir: Path, vector_store: VectorStore, embedder: QwenEmbedding
    ) -> None:
        service = IngestionService(vector_store=vector_store, embedder=embedder)
        first = service.run(sample_documents_dir)

        second = service.run(sample_documents_dir)

        assert second.documents_loaded == 4
        assert second.new_documents == 0
        assert second.duplicates_skipped == 4
        assert second.chunks_created == 0
        assert vector_store.count() == first.chunks_created  # unchanged

    def test_adding_one_new_file_only_ingests_that_file(
        self, sample_documents_dir: Path, vector_store: VectorStore, embedder: QwenEmbedding
    ) -> None:
        service = IngestionService(vector_store=vector_store, embedder=embedder)
        service.run(sample_documents_dir)

        (sample_documents_dir / "extra.txt").write_text(
            "Support hours are 9am to 6pm Eastern, Monday through Friday.", encoding="utf-8"
        )

        result = service.run(sample_documents_dir)

        assert result.new_documents == 1
        assert result.duplicates_skipped == 4

    def test_empty_directory_produces_zero_result(
        self, tmp_path: Path, vector_store: VectorStore, embedder: QwenEmbedding
    ) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        service = IngestionService(vector_store=vector_store, embedder=embedder)

        result = service.run(empty_dir)

        assert result.documents_found == 0
        assert result.status == "success"

    def test_missing_directory_raises_file_not_found(
        self, tmp_path: Path, vector_store: VectorStore, embedder: QwenEmbedding
    ) -> None:
        service = IngestionService(vector_store=vector_store, embedder=embedder)
        with pytest.raises(FileNotFoundError):
            service.run(tmp_path / "does-not-exist")

    def test_embedding_failure_is_captured_not_raised(
        self,
        sample_documents_dir: Path,
        vector_store: VectorStore,
        embedder: QwenEmbedding,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def broken_embed_documents(chunks):  # noqa: ANN001
            raise RuntimeError("simulated embedding outage")

        monkeypatch.setattr(embedder, "embed_documents", broken_embed_documents)
        service = IngestionService(vector_store=vector_store, embedder=embedder)

        result = service.run(sample_documents_dir)

        assert result.status == "error"
        assert result.errors
        assert result.vectors_stored == 0

    def test_documents_are_registered_with_correct_chunk_counts(
        self, sample_documents_dir: Path, vector_store: VectorStore, embedder: QwenEmbedding
    ) -> None:
        service = IngestionService(vector_store=vector_store, embedder=embedder)
        service.run(sample_documents_dir)

        registered = vector_store.list_ingested_documents()
        assert len(registered) == 4
        assert all(doc["chunk_count"] > 0 for doc in registered)
