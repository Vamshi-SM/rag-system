"""Shared ingestion pipeline service.

This is the single implementation of the Load -> Dedup -> Chunk ->
Embed -> Store pipeline (Phase 6). Both ``scripts/ingest.py`` (CLI) and
``src/api/routes.py`` (``POST /ingest``) call into
:class:`IngestionService` rather than each having their own copy of
this logic, per the "never duplicate code" requirement.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.chunking.chunker import Chunk, DocumentChunker
from src.embeddings.base_embedding import BaseEmbedding
from src.loaders.base_loader import LoadedDocument
from src.loaders.document_loader import DocumentLoader
from src.utils.logger import get_logger
from src.vectordb.vector_store import VectorStore

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Structured summary of a single ingestion run.

    Used both to print the CLI's progress output and to build the
    ``POST /ingest`` JSON response, so the two surfaces can never
    disagree about what actually happened.
    """

    documents_found: int = 0
    documents_loaded: int = 0
    duplicates_skipped: int = 0
    new_documents: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    vectors_stored: int = 0
    total_vectors_in_store: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """"success" if nothing went wrong, "partial" if some errors
        occurred but the run still completed, "error" if the whole run failed."""
        if self.errors and self.vectors_stored == 0 and self.new_documents > 0:
            return "error"
        return "partial" if self.errors else "success"


def compute_checksum(file_path: Path) -> str:
    """Compute a SHA-256 checksum of a file's contents.

    Used to detect duplicate/unchanged documents so re-running
    ingestion is idempotent rather than re-inserting the same content.
    """
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class IngestionService:
    """Runs the full ingestion pipeline against a configured data directory.

    Args:
        vector_store: The (already-initialized) Phase 5 vector store to
            write into and check for duplicates against.
        embedder: The Phase 4 embedding provider to use.
        loader: The Phase 2 document loader. Defaults to a fresh
            ``DocumentLoader()`` if not provided.
        chunker: The Phase 3 chunker. Defaults to a
            ``DocumentChunker`` configured from ``src.config.settings``
            if not provided.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: BaseEmbedding,
        loader: DocumentLoader | None = None,
        chunker: DocumentChunker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.loader = loader or DocumentLoader()

        if chunker is None:
            from src.config import settings

            chunker = DocumentChunker(
                chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
            )
        self.chunker = chunker

    def _deduplicate(
        self, documents: list[LoadedDocument]
    ) -> tuple[list[LoadedDocument], int]:
        """Split loaded documents into "new" vs "already ingested".

        New documents have their ``id`` overwritten with their content
        checksum so chunk-to-document linkage is stable across runs.
        """
        new_documents: list[LoadedDocument] = []
        skipped_count = 0

        for document in documents:
            source = document.get("metadata", {}).get("source")
            if not source:
                logger.warning(
                    "Document '%s' has no source path; skipping dedup check",
                    document["filename"],
                )
                new_documents.append(document)
                continue

            try:
                checksum = compute_checksum(Path(source))
            except OSError as exc:
                logger.error("Could not read '%s' to compute checksum: %s", source, exc)
                continue

            existing_document_id = self.vector_store.is_document_ingested(checksum)
            if existing_document_id:
                logger.info(
                    "Skipping duplicate document '%s' (already ingested as %s)",
                    document["filename"],
                    existing_document_id,
                )
                skipped_count += 1
                continue

            document["id"] = checksum
            document["metadata"]["checksum"] = checksum
            new_documents.append(document)

        return new_documents, skipped_count

    def _register(self, documents: list[LoadedDocument], chunks: list[Chunk]) -> None:
        """Record each successfully-ingested document in the registry."""
        chunk_counts: dict[str, int] = {}
        for chunk in chunks:
            chunk_counts[chunk["document_id"]] = chunk_counts.get(chunk["document_id"], 0) + 1

        for document in documents:
            document_id = document["id"]
            chunk_count = chunk_counts.get(document_id, 0)
            if chunk_count == 0:
                continue
            self.vector_store.register_document(
                document_id=document_id,
                checksum=document["metadata"].get("checksum", document_id),
                filename=document["filename"],
                source=document["metadata"].get("source"),
                chunk_count=chunk_count,
            )

    def run(self, data_dir: str | Path) -> IngestionResult:
        """Run the full ingestion pipeline against ``data_dir``.

        Never raises for expected failure modes (missing files, empty
        directory, embedding failures) - those are captured in
        ``IngestionResult.errors`` and reflected in ``.status``.
        Raises only for truly unexpected conditions (e.g. the data
        directory itself doesn't exist), which callers should treat as
        a 4xx/validation error rather than a partial success.

        Args:
            data_dir: Directory to recursively load documents from.

        Returns:
            An :class:`IngestionResult` summarizing what happened.

        Raises:
            FileNotFoundError: If ``data_dir`` does not exist.
        """
        start_time = time.time()
        result = IngestionResult()
        data_dir = Path(data_dir)

        # --- Stage 1: Load ---
        documents = self.loader.load_directory(data_dir)
        result.documents_found = len(documents)
        result.documents_loaded = len(documents)

        if not documents:
            logger.warning("No documents found in '%s'; nothing to ingest.", data_dir)
            result.elapsed_seconds = time.time() - start_time
            return result

        # --- Stage 2: Deduplicate / incremental ingestion ---
        new_documents, skipped_count = self._deduplicate(documents)
        result.duplicates_skipped = skipped_count
        result.new_documents = len(new_documents)

        if not new_documents:
            logger.info("All documents already ingested; nothing new to process.")
            result.total_vectors_in_store = self.vector_store.count()
            result.elapsed_seconds = time.time() - start_time
            return result

        # --- Stage 3: Chunk ---
        chunks = self.chunker.chunk_documents(new_documents)
        result.chunks_created = len(chunks)

        if not chunks:
            logger.warning("No chunks produced from new documents; nothing to embed or store.")
            result.total_vectors_in_store = self.vector_store.count()
            result.elapsed_seconds = time.time() - start_time
            return result

        # --- Stage 4: Embed ---
        try:
            embedded_chunks = self.embedder.embed_documents(chunks)
        except Exception as exc:  # noqa: BLE001 - never let embedding crash ingestion
            logger.exception("Embedding stage failed: %s", exc)
            result.errors.append(f"embedding_failed: {exc}")
            embedded_chunks = []

        result.embeddings_generated = len(embedded_chunks)

        if not embedded_chunks:
            logger.error("No embeddings were generated; aborting before database write.")
            if not result.errors:
                result.errors.append("no_embeddings_generated")
            result.total_vectors_in_store = self.vector_store.count()
            result.elapsed_seconds = time.time() - start_time
            return result

        # --- Stage 5: Store ---
        inserted = self.vector_store.insert_many(embedded_chunks)
        result.vectors_stored = inserted

        self._register(new_documents, chunks)

        result.total_vectors_in_store = self.vector_store.count()
        result.elapsed_seconds = time.time() - start_time

        logger.info(
            "Ingestion complete: %d documents loaded, %d new, %d duplicates skipped, "
            "%d chunks, %d embeddings, %d vectors stored (%.2fs)",
            result.documents_loaded,
            result.new_documents,
            result.duplicates_skipped,
            result.chunks_created,
            result.embeddings_generated,
            result.vectors_stored,
            result.elapsed_seconds,
        )
        return result
