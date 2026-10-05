#!/usr/bin/env python3
"""Ingestion CLI: Read -> Extract -> Chunk -> Embed -> Store.

This is a thin wrapper around ``src.ingestion.ingestion_service.IngestionService``
- the same service class used by ``POST /ingest`` in the Phase 9 API - so
the CLI and the API can never drift out of sync on ingestion behavior.

Incremental and idempotent: documents whose content hasn't changed
since the last run are skipped (detected via a SHA-256 checksum of the
file contents), so re-running this script only ingests what's new.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --data-dir data/documents
    python scripts/ingest.py --reset      # wipe the store and re-ingest everything
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as `python scripts/ingest.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.embeddings.factory import build_embedder
from src.ingestion.ingestion_service import IngestionService
from src.utils.logger import get_logger
from src.vectordb.vector_store import VectorStore

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG ingestion pipeline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.data_directory,
        help="Directory to recursively load documents from.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all existing vectors and the document registry before ingesting.",
    )
    return parser.parse_args()


def print_result(result) -> None:
    """Print the ingestion result in the required console format."""
    print(f"{result.documents_loaded} documents loaded\n")

    if result.duplicates_skipped:
        print(f"{result.duplicates_skipped} duplicate document(s) skipped (already ingested)\n")

    if result.new_documents == 0:
        print("Completed successfully. (no new documents)")
        return

    print("Chunking...")
    print(f"{result.chunks_created} chunks created\n")

    if result.chunks_created == 0:
        print("Completed successfully. (no chunks produced)")
        return

    print("Generating embeddings...")
    print(f"{result.embeddings_generated} embeddings generated\n")

    if result.embeddings_generated == 0:
        print("Error: embedding generation failed for all chunks. See logs for details.")
        return

    print("Saving vectors...")
    print(f"{result.vectors_stored} vectors stored\n")
    print(f"Total vectors in store: {result.total_vectors_in_store}")

    if result.errors:
        print(f"Completed with warnings: {'; '.join(result.errors)}")
    else:
        print("Completed successfully.")


def main() -> None:
    args = parse_args()
    settings.ensure_directories()
    pipeline_start = time.time()

    vector_store = VectorStore(database_path=settings.database_path, default_top_k=settings.top_k)
    embedder = build_embedder(settings)

    try:
        if args.reset:
            print("Resetting existing vector store...")
            vector_store.delete_all()

        print("Loading documents...")
        service = IngestionService(vector_store=vector_store, embedder=embedder)

        try:
            result = service.run(args.data_dir)
        except FileNotFoundError as exc:
            logger.error("Ingestion aborted: %s", exc)
            print(f"Error: {exc}")
            return

        print_result(result)

    except Exception as exc:  # noqa: BLE001 - top-level safety net; never crash silently
        logger.exception("Ingestion pipeline failed unexpectedly: %s", exc)
        print(f"Error: ingestion failed unexpectedly ({exc}). See logs for details.")
    finally:
        vector_store.close()
        logger.info("Ingestion pipeline finished in %.2fs", time.time() - pipeline_start)


if __name__ == "__main__":
    main()
