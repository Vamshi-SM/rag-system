"""
High-level vector store API backed by PostgreSQL + pgvector.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from psycopg.types.json import Json

from src.embeddings.base_embedding import EmbeddedChunk
from src.utils.logger import get_logger
from src.vectordb.base_store import BaseVectorStore
from src.vectordb.postgres_database import PostgresDatabase

logger = get_logger(__name__)


class SearchResult(TypedDict):
    id: str
    chunk_id: str
    document_id: str | None
    text: str
    metadata: dict[str, Any]
    distance: float
    similarity: float


class VectorStoreError(Exception):
    pass


class PostgresStore(BaseVectorStore):

    def __init__(
        self,
        database_url: str,
        default_top_k: int = 5,
    ):
        self.database = PostgresDatabase(database_url)
        self.default_top_k = default_top_k
        self._dimensions: int | None = None

        self.initialize_database()

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    def initialize_database(self) -> None:

        self.database.execute(
            """
            CREATE EXTENSION IF NOT EXISTS vector;
            """
        )

        self.database.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (

                id TEXT PRIMARY KEY,

                chunk_id TEXT NOT NULL,

                document_id TEXT,

                text TEXT NOT NULL,

                metadata JSONB,

                embedding VECTOR(768)

            );
            """
        )

        self.database.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_document
            ON chunks(document_id);
            """
        )

        logger.info("Initialized PostgreSQL schema")

    # --------------------------------------------------------
    # Writes
    # --------------------------------------------------------

    def insert_document(
        self,
        embedded_chunk: EmbeddedChunk,
    ) -> None:

        self.insert_many([embedded_chunk])

    def insert_many(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> int:

        if not embedded_chunks:
            return 0

        if self._dimensions is None:
            self._dimensions = len(
                embedded_chunks[0]["embedding"]
            )

        inserted = 0

        with self.database.connection.cursor() as cur:

            for chunk in embedded_chunks:

                metadata = chunk.get("metadata", {})

                chunk_id = metadata.get("chunk_id")

                document_id = metadata.get("document_id")

                row_id = metadata.get("id") or chunk_id

                if len(chunk["embedding"]) != self._dimensions:
                    logger.warning(
                        "Skipping chunk %s because embedding dimensions mismatch",
                        chunk_id,
                    )
                    continue

                cur.execute(
                    """
                    INSERT INTO chunks
                    (
                        id,
                        chunk_id,
                        document_id,
                        text,
                        metadata,
                        embedding
                    )
                    VALUES
                    (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )

                    ON CONFLICT(id)

                    DO UPDATE SET

                        chunk_id = EXCLUDED.chunk_id,

                        document_id = EXCLUDED.document_id,

                        text = EXCLUDED.text,

                        metadata = EXCLUDED.metadata,

                        embedding = EXCLUDED.embedding;
                    """,
                    (
                        row_id,
                        chunk_id,
                        document_id,
                        chunk["text"],
                        Json(metadata),
                        chunk["embedding"],
                    ),
                )

                inserted += 1

        logger.info(
            "Inserted %d/%d chunks",
            inserted,
            len(embedded_chunks),
        )

        return inserted
    