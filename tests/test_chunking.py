"""Unit tests for src/chunking (Phase 3)."""

from __future__ import annotations

import uuid

from src.chunking.chunker import DocumentChunker


def _make_document(text: str, page_texts: list[str] | None = None) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "filename": "sample.txt",
        "text": text,
        "metadata": {
            "extension": ".txt",
            "pages": None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "source": "/tmp/sample.txt",
            "page_texts": page_texts,
        },
    }


class TestDocumentChunker:
    def test_chunk_document_produces_chunks_with_required_fields(self) -> None:
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=5)
        document = _make_document("This is a reasonably long piece of text. " * 10)

        chunks = chunker.chunk_document(document)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk["document_id"] == document["id"]
            assert chunk["chunk_id"]
            assert chunk["text"].strip()
            assert chunk["metadata"]["filename"] == "sample.txt"

    def test_chunk_ids_are_unique(self) -> None:
        chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
        document = _make_document("word " * 100)

        chunks = chunker.chunk_document(document)
        chunk_ids = [c["chunk_id"] for c in chunks]

        assert len(chunk_ids) == len(set(chunk_ids))

    def test_empty_document_produces_no_chunks(self) -> None:
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        document = _make_document("   ")

        assert chunker.chunk_document(document) == []

    def test_page_texts_produce_per_page_metadata(self) -> None:
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        document = _make_document(
            "page one text\n\npage two text",
            page_texts=["page one text", "page two text"],
        )

        chunks = chunker.chunk_document(document)

        pages = sorted({c["metadata"]["page"] for c in chunks})
        assert pages == [1, 2]

    def test_no_page_texts_yields_none_page(self) -> None:
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        document = _make_document("just some plain text with no pagination")

        chunks = chunker.chunk_document(document)

        assert all(c["metadata"]["page"] is None for c in chunks)

    def test_chunk_documents_flattens_multiple_documents(self) -> None:
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        documents = [_make_document("first document text"), _make_document("second document text")]

        chunks = chunker.chunk_documents(documents)

        document_ids = {c["document_id"] for c in chunks}
        assert document_ids == {documents[0]["id"], documents[1]["id"]}

    def test_rejects_overlap_larger_than_chunk_size(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=100)
