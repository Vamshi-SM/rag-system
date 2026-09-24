"""Low-level PostgreSQL connection management with pgvector.

This module owns exactly one responsibility: handing the vector DB
layer ready-to-use ``psycopg`` connections backed by a
``psycopg_pool.ConnectionPool`` so concurrent API requests never
serialize on a single shared connection.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PostgresDatabase:
    """Owns the psycopg connection pool and per-connection pgvector setup.

    Args:
        database_url: PostgreSQL connection string, e.g.
            ``postgresql://user:password@host:5432/dbname``.
        pool_size: Maximum number of pooled connections. A connection is
            checked out only for the duration of a single query, so a
            size of ~2x the expected concurrent request count is a safe
            ceiling (each Cloud Run / worker process gets its own pool).

    Concurrency note:
        Every ``query`` / ``execute`` / ``transaction`` call checks a
        connection out of the pool and returns it on exit, so multiple
        worker threads can hit the database simultaneously. The pool
        blocks (up to its timeout) when exhausted rather than erroring.
    """

    def __init__(self, database_url: str, pool_size: int = 10):
        self.pool = ConnectionPool(
            database_url,
            min_size=1,
            max_size=max(1, int(pool_size)),
            timeout=30,
            kwargs={"row_factory": dict_row, "autocommit": True},
            # register_vector must be applied to every *new* physical
            # connection, not just the first one - this callback runs
            # whenever the pool opens a fresh connection.
            configure=self._configure_connection,
            open=True,
        )
        logger.info("Connected to PostgreSQL (pool max_size=%d)", pool_size)

    @staticmethod
    def _configure_connection(conn: psycopg.Connection) -> None:
        register_vector(conn)

    @property
    def fts_available(self) -> bool:
        """Whether the tsvector full-text index is available for hybrid search."""
        return self._fts_available

    def initialize_database(self) -> None:
        """Create the extensions, tables, indexes, and FTS column.

        The ``tsv`` generated column (PG 12+) mirrors SQLite's FTS5
        index: it stays in sync with ``text`` automatically and powers
        keyword search for hybrid retrieval. If the server is too old
        to support generated columns we degrade gracefully to
        pure-vector search by flipping ``fts_available``.
        """
        self._fts_available = True
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE EXTENSION IF NOT EXISTS vector;

                    CREATE TABLE IF NOT EXISTS chunks (
                        id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB,
                        embedding VECTOR(768)
                    );

                    CREATE TABLE IF NOT EXISTS document_registry (
                        document_id TEXT PRIMARY KEY,
                        checksum TEXT UNIQUE,
                        filename TEXT,
                        source TEXT,
                        chunk_count INTEGER,
                        ingested_at TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                    ON chunks
                    USING hnsw (embedding vector_cosine_ops);
                """)
                try:
                    cur.execute("""
                        ALTER TABLE chunks
                        ADD COLUMN IF NOT EXISTS tsv tsvector
                        GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;

                        CREATE INDEX IF NOT EXISTS chunks_tsv_idx
                        ON chunks USING GIN (tsv);
                    """)
                except psycopg.Error as exc:
                    self._fts_available = False
                    logger.warning(
                        "Full-text search setup failed (%s); hybrid search "
                        "disabled, falling back to pure-vector retrieval",
                        exc,
                    )
        logger.info("Database schema initialized")

    def execute(self, query: str, params: tuple | None = None) -> None:
        """Execute a write query on a pooled connection."""
        with self.pool.connection() as conn:
            conn.execute(query, params)

    def query(self, query: str, params: tuple | None = None) -> list[dict]:
        """Execute a read query on a pooled connection and return all rows."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()

    @contextmanager
    def transaction(self) -> Iterator[psycopg.Connection]:
        """Run several statements as a single atomic transaction on one
        pooled connection.

        The pool runs in autocommit mode for single statements, but
        ``conn.transaction()`` still issues explicit BEGIN/COMMIT, so
        multi-statement atomicity (used by ``insert_many``) is preserved.
        """
        with self.pool.connection() as conn:
            with conn.transaction():
                yield conn

    def close(self) -> None:
        """Close all pooled connections."""
        if self.pool:
            self.pool.close()
        logger.info("PostgreSQL connection pool closed")
