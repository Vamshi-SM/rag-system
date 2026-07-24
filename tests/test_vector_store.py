"""Unit tests for src/vectordb (Phase 5, plus the Phase 6 document registry)."""

from __future__ import annotations

import pytest

from src.vectordb.vector_store import VectorStore, VectorStoreError


def _vec(base: list[float]) -> list[float]:
    """Pad a small test vector to 768 dimensions to match the schema."""
    return base + [0.0] * (768 - len(base))

def _make_chunk(chunk_id: str, document_id: str, text: str, embedding: list[float]) -> dict:
    return {
        "text": text,
        "embedding": _vec(embedding),  # Pad the embedding here
        "metadata": {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "id": chunk_id,
            "filename": f"{document_id}.txt",
            "page": None,
        },
    }


class TestVectorStoreCRUD:
    def test_insert_and_count(self, vector_store: VectorStore) -> None:
        chunk = _make_chunk("c1", "d1", "hello world", [1.0, 0.0, 0.0])
        inserted = vector_store.insert_many([chunk])
        assert inserted == 1
        assert vector_store.count() == 1

    def test_insert_many(self, vector_store: VectorStore) -> None:
        chunks = [
            _make_chunk(f"c{i}", "d1", f"text {i}", [float(i), 0.0, 0.0]) for i in range(5)
        ]
        inserted = vector_store.insert_many(chunks)
        assert inserted == 5
        assert vector_store.count() == 5

    def test_insert_many_empty_list_returns_zero(self, vector_store: VectorStore) -> None:
        assert vector_store.insert_many([]) == 0
        assert vector_store.count() == 0

    def test_delete_document_removes_only_its_chunks(self, vector_store: VectorStore) -> None:
        vector_store.insert_many(
            [
                _make_chunk("c1", "d1", "doc1 text", [1.0, 0.0, 0.0]),
                _make_chunk("c2", "d2", "doc2 text", [0.0, 1.0, 0.0]),
            ]
        )
        deleted = vector_store.delete_document("d1")
        assert deleted == 1
        assert vector_store.count() == 1

    def test_delete_all_clears_store(self, vector_store: VectorStore) -> None:
        vector_store.insert_many(
            [_make_chunk("c1", "d1", "text", [1.0, 0.0, 0.0])]
        )
        vector_store.delete_all()
        assert vector_store.count() == 0

    def test_count_on_empty_store(self, vector_store: VectorStore) -> None:
        assert vector_store.count() == 0


class TestSimilaritySearch:
    def test_exact_match_has_similarity_near_one(self, vector_store: VectorStore) -> None:
        vector_store.insert_many([_make_chunk("c1", "d1", "target", [1.0, 0.0, 0.0])])
        # Wrap query in _vec()
        results = vector_store.similarity_search(_vec([1.0, 0.0, 0.0]), top_k=1) 
        assert len(results) == 1
        assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-4)

    def test_results_ordered_by_similarity_descending(self, vector_store: VectorStore) -> None:
        vector_store.insert_many(
            [
                _make_chunk("close", "d1", "close match", [0.9, 0.1, 0.0]),
                _make_chunk("far", "d1", "far match", [0.0, 0.0, 1.0]),
                _make_chunk("exact", "d1", "exact match", [1.0, 0.0, 0.0]),
            ]
        )
        # Wrap query in _vec()
        results = vector_store.similarity_search(_vec([1.0, 0.0, 0.0]), top_k=3)
        similarities = [r["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True)
        assert results[0]["chunk_id"] == "exact"

    def test_top_k_limits_result_count(self, vector_store: VectorStore) -> None:
        vector_store.insert_many(
            [_make_chunk(f"c{i}", "d1", f"text {i}", [float(i), 0.0, 1.0]) for i in range(10)]
        )
        # Wrap query in _vec()
        results = vector_store.similarity_search(_vec([0.0, 0.0, 1.0]), top_k=3)
        assert len(results) == 3

    def test_similarity_search_on_empty_store_returns_empty(self, vector_store: VectorStore) -> None:
        # Wrap query in _vec()
        assert vector_store.similarity_search(_vec([1.0, 0.0, 0.0])) == []


    def test_dimension_mismatch_raises(self, vector_store: VectorStore) -> None:
        vector_store.insert_many([_make_chunk("c1", "d1", "text", [1.0, 0.0, 0.0])])
        with pytest.raises(VectorStoreError):
            # Intentionally NOT padded to trigger the error
            vector_store.similarity_search([1.0, 0.0])
            
    def test_metadata_round_trips_through_json(self, vector_store: VectorStore) -> None:
        chunk = _make_chunk("c1", "d1", "text", [1.0, 0.0, 0.0])
        chunk["metadata"]["page"] = 7
        vector_store.insert_many([chunk])
        results = vector_store.similarity_search([1.0, 0.0, 0.0], top_k=1)
        assert results[0]["metadata"]["page"] == 7


class TestDocumentRegistry:
    def test_is_document_ingested_false_for_unknown_checksum(
        self, vector_store: VectorStore
    ) -> None:
        assert vector_store.is_document_ingested("nonexistent-checksum") is None

    def test_register_then_check(self, vector_store: VectorStore) -> None:
        vector_store.register_document("doc-1", "checksum-abc", "file.txt", "/tmp/file.txt", 3)
        assert vector_store.is_document_ingested("checksum-abc") == "doc-1"

    def test_list_ingested_documents(self, vector_store: VectorStore) -> None:
        vector_store.register_document("doc-1", "sum-1", "a.txt", "/tmp/a.txt", 2)
        vector_store.register_document("doc-2", "sum-2", "b.txt", "/tmp/b.txt", 4)
        documents = vector_store.list_ingested_documents()
        assert len(documents) == 2
        assert {d["document_id"] for d in documents} == {"doc-1", "doc-2"}

    def test_delete_document_also_unregisters(self, vector_store: VectorStore) -> None:
        vector_store.insert_many([_make_chunk("c1", "d1", "text", [1.0, 0.0, 0.0])])
        vector_store.register_document("d1", "checksum-d1", "d1.txt", "/tmp/d1.txt", 1)
        vector_store.delete_document("d1")
        assert vector_store.is_document_ingested("checksum-d1") is None

    def test_delete_all_clears_registry(self, vector_store: VectorStore) -> None:
        vector_store.register_document("doc-1", "sum-1", "a.txt", "/tmp/a.txt", 2)
        vector_store.delete_all()
        assert vector_store.list_ingested_documents() == []


class TestPersistence:
    def test_dimensions_recovered_after_reconnect(self, db_path) -> None:
        store1 = VectorStore(database_path=db_path)
        store1.insert_many([_make_chunk("c1", "d1", "text", [1.0, 0.0, 0.0])])
        store1.close()

        store2 = VectorStore(database_path=db_path)
        try:
            assert store2.count() == 1
            # Wrap query in _vec()
            results = store2.similarity_search(_vec([1.0, 0.0, 0.0]), top_k=1)
            assert len(results) == 1
        finally:
            store2.close()
