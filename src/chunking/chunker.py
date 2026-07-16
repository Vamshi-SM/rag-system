"""Text chunking module (Phase 3).

Splits loaded documents into overlapping chunks sized in
*approximate tokens*, using LangChain's ``RecursiveCharacterTextSplitter``
configured with a token-aware length function (via ``tiktoken`` when
available, falling back to a simple whitespace heuristic otherwise so
the module never hard-depends on a specific tokenizer being installed).
"""

from __future__ import annotations

import uuid
from typing import Callable, TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.loaders.base_loader import LoadedDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChunkMetadata(TypedDict, total=False):
    """Metadata carried on every chunk."""

    page: int | None
    filename: str
    extension: str
    source: str


class Chunk(TypedDict):
    """The canonical shape returned by the chunker."""

    chunk_id: str
    document_id: str
    text: str
    metadata: ChunkMetadata


def _build_token_length_function() -> Callable[[str], int]:
    """Return a function estimating token count for a string.

    Prefers ``tiktoken`` (accurate, matches most modern LLM tokenizers
    closely enough for chunk-sizing purposes). Falls back to a
    words-based heuristic (~0.75 tokens per word inverted) if
    ``tiktoken`` isn't installed, so the chunker degrades gracefully
    rather than crashing.
    """
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")

        def _token_len(text: str) -> int:
            return len(encoding.encode(text))

        return _token_len
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "tiktoken unavailable (%s); falling back to word-count token estimate", exc
        )

        def _approx_token_len(text: str) -> int:
            # Rough heuristic: 1 token ~= 0.75 words -> words / 0.75
            word_count = len(text.split())
            return int(word_count / 0.75) if word_count else 0

        return _approx_token_len


class DocumentChunker:
    """Splits documents into overlapping, metadata-preserving chunks.

    Args:
        chunk_size: Target chunk size in approximate tokens.
        chunk_overlap: Overlap between consecutive chunks, in approximate tokens.
    """

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._token_length_function = _build_token_length_function()

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_length_function,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _make_chunk(
        self, document: LoadedDocument, raw_text: str, page: int | None
    ) -> Chunk:
        metadata = document.get("metadata", {})
        return {
            "chunk_id": str(uuid.uuid4()),
            "document_id": document["id"],
            "text": raw_text,
            "metadata": {
                "page": page,
                "filename": document["filename"],
                "extension": metadata.get("extension", ""),
                "source": metadata.get("source", ""),
            },
        }

    def chunk_document(self, document: LoadedDocument) -> list[Chunk]:
        """Split a single loaded document into chunks.

        If the loader captured per-page text (currently PDFs only),
        each page is chunked independently so every chunk carries an
        accurate page number. Otherwise the whole document is treated
        as one continuous text stream with ``page=None``.
        """
        metadata = document.get("metadata", {})
        page_texts = metadata.get("page_texts")

        chunks: list[Chunk] = []

        if page_texts:
            for page_number, page_text in enumerate(page_texts, start=1):
                if not page_text or not page_text.strip():
                    continue
                for raw_chunk in self._splitter.split_text(page_text):
                    if raw_chunk.strip():
                        chunks.append(self._make_chunk(document, raw_chunk, page_number))
        else:
            text = document["text"]
            if text and text.strip():
                for raw_chunk in self._splitter.split_text(text):
                    if raw_chunk.strip():
                        chunks.append(self._make_chunk(document, raw_chunk, None))

        logger.info(
            "Chunked '%s' into %d chunks (chunk_size=%d, overlap=%d)",
            document["filename"],
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks

    def chunk_documents(self, documents: list[LoadedDocument]) -> list[Chunk]:
        """Split a list of loaded documents into a flat list of chunks."""
        all_chunks: list[Chunk] = []
        for document in documents:
            all_chunks.extend(self.chunk_document(document))

        logger.info(
            "Chunked %d documents into %d total chunks", len(documents), len(all_chunks)
        )
        return all_chunks
