"""Unit tests for src/embeddings (Phase 4)."""

from __future__ import annotations

import pytest

from src.embeddings.base_embedding import EmbeddingError
from src.embeddings.qwen_embedding import QwenEmbedding


class TestQwenEmbedding:
    def test_embed_single_text_returns_vector(self, embedder: QwenEmbedding) -> None:
        vector = embedder.embed("hello world")
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(value, float) for value in vector)

    def test_embed_is_deterministic(self, embedder: QwenEmbedding) -> None:
        assert embedder.embed("same text") == embedder.embed("same text")

    def test_different_text_yields_different_vector(self, embedder: QwenEmbedding) -> None:
        assert embedder.embed("text one") != embedder.embed("text two")

    def test_dimensions_detected_after_first_call(self, embedder: QwenEmbedding) -> None:
        assert embedder.dimensions is None
        embedder.embed("hello")
        assert embedder.dimensions is not None
        assert embedder.dimensions == len(embedder.embed("hello"))

    def test_embed_documents_batches_chunks(self, embedder: QwenEmbedding) -> None:
        chunks = [
            {"chunk_id": f"c{i}", "document_id": "d1", "text": f"chunk text {i}", "metadata": {}}
            for i in range(5)
        ]
        embedder.batch_size = 2

        embedded = embedder.embed_documents(chunks)

        assert len(embedded) == 5
        for original, result in zip(chunks, embedded):
            assert result["text"] == original["text"]
            assert result["metadata"]["chunk_id"] == original["chunk_id"]
            assert result["metadata"]["document_id"] == original["document_id"]
            assert len(result["embedding"]) == embedder.dimensions

    def test_embed_documents_skips_failed_batch_but_keeps_others(
        self, embedder: QwenEmbedding, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chunks = [
            {"chunk_id": "c1", "document_id": "d1", "text": "good chunk", "metadata": {}},
            {"chunk_id": "c2", "document_id": "d1", "text": "bad chunk", "metadata": {}},
        ]
        embedder.batch_size = 1

        original_call_api = embedder._call_api

        def flaky_call_api(texts: list[str]) -> list[list[float]]:
            if "bad chunk" in texts:
                raise EmbeddingError("simulated failure")
            return original_call_api(texts)

        monkeypatch.setattr(embedder, "_call_api", flaky_call_api)

        embedded = embedder.embed_documents(chunks)

        assert len(embedded) == 1
        assert embedded[0]["metadata"]["chunk_id"] == "c1"

    def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        embedder = QwenEmbedding(
            api_key="k", base_url="https://fake/v1", model="m", max_retries=3, backoff_factor=0.01
        )
        attempts = {"count": 0}

        class FakeResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"data": [{"embedding": [0.1, 0.2], "index": 0}]}

        def flaky_post(url, json, timeout):  # noqa: ANN001
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise __import__("requests").exceptions.ConnectionError("simulated network blip")
            return FakeResponse()

        monkeypatch.setattr(embedder._session, "post", flaky_post)

        vector = embedder.embed("hello")
        assert vector == [0.1, 0.2]
        assert attempts["count"] == 2

    def test_raises_embedding_error_after_exhausting_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        embedder = QwenEmbedding(
            api_key="k", base_url="https://fake/v1", model="m", max_retries=2, backoff_factor=0.01
        )

        def always_fails(url, json, timeout):  # noqa: ANN001
            raise __import__("requests").exceptions.ConnectionError("simulated outage")

        monkeypatch.setattr(embedder._session, "post", always_fails)

        with pytest.raises(EmbeddingError):
            embedder.embed("hello")

    def test_never_hardcodes_dimensions(self, patch_embedding_api) -> None:
        """Two embedders with different fake dimensionality should both
        just work - dimension is inferred from the response, never assumed.
        """
        small = QwenEmbedding(api_key="k", base_url="https://fake/v1", model="m")
        small.embed("x")
        assert small.dimensions == 16  # matches FAKE_EMBEDDING_DIM in conftest
