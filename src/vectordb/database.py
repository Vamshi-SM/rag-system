"""Low-level SQLite connection management with the sqlite-vec extension.

This module owns exactly one responsibility: giving the rest of the
vector DB layer a ready-to-use ``sqlite3.Connection`` that has the
``sqlite-vec`` extension loaded and the base schema created.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import sqlite_vec

from src.utils.logger import get_logger
from src.vectordb.schema import (
    CREATE_CHUNKS_FTS_SQL,
    CREATE_CHUNKS_TABLE_SQL,
    CREATE_DB_META_TABLE_SQL,
    CREATE_DOCUMENT_REGISTRY_TABLE_SQL,
    CREATE_INDEX_CHUNK_ID_SQL,
    CREATE_INDEX_DOCUMENT_ID_SQL,
    CREATE_INDEX_DOCUMENT_REGISTRY_CHECKSUM_SQL,
)

logger = get_logger(__name__)


class Database:
    """Owns the raw SQLite connection and sqlite-vec extension loading.

    Args:
        database_path: File path where the SQLite database lives.
            Created automatically (including parent directories) if it
            doesn't exist yet.

    Note on concurrency:
        A single connection is shared for the lifetime of this object.
        The Phase 9 API serves each request's blocking DB calls from a
        worker thread pool (via ``starlette.concurrency.run_in_threadpool``),
        so the connection is opened with ``check_same_thread=False`` and
        every access is serialized with an internal lock. SQLite only
        supports one writer at a time anyway, so this doesn't sacrifice
        real concurrency - it just makes cross-thread access safe.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self.connection = sqlite3.connect(str(self.database_path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._load_vec_extension()
        self._apply_pragmas()
        self._fts_available = True

        logger.info("Connected to SQLite database at '%s'", self.database_path)

    def _load_vec_extension(self) -> None:
        """Load the sqlite-vec extension into this connection."""
        self.connection.enable_load_extension(True)
        sqlite_vec.load(self.connection)
        self.connection.enable_load_extension(False)
        logger.debug("sqlite-vec extension loaded (version=%s)", sqlite_vec.__version__)

    def _apply_pragmas(self) -> None:
        """Apply production-oriented SQLite tuning pragmas to the connection.

        * ``journal_mode=WAL`` — write-ahead logging allows concurrent readers
          while a writer holds the lock, and is dramatically faster than the
          default rollback journal for our read-heavy query workload.
        * ``synchronous=NORMAL`` — the safe pairing with WAL; one fsync per
          transaction instead of per write. (Never ``OFF`` — that risks
          corruption on power loss.)
        * ``busy_timeout`` — wait up to 5s for a locked DB instead of failing
          immediately under contention.
        * ``temp_store=MEMORY`` / ``mmap_size`` / ``cache_size`` — keep
          temporary tables and the page cache in RAM and memory-map the DB
          file for faster reads.
        """
        with self._lock:
            for pragma, value in (
                ("journal_mode", "WAL"),
                ("synchronous", "NORMAL"),
                ("busy_timeout", "5000"),
                ("temp_store", "MEMORY"),
                ("mmap_size", "268435456"),
                ("cache_size", "-65536"),
            ):
                self.connection.execute(f"PRAGMA {pragma}={value}")

    @property
    def fts_available(self) -> bool:
        """Whether the FTS5 full-text index is available for hybrid search."""
        return self._fts_available

    def initialize_database(self) -> None:
        """Create the base (non-vector) schema if it doesn't already exist.

        The vec0 virtual table is intentionally NOT created here since
        its dimensionality isn't known until the first embedding is
        produced - see :class:`src.vectordb.vector_store.VectorStore`.
        """
        with self._lock, self.connection:
            self.connection.execute(CREATE_CHUNKS_TABLE_SQL)
            self.connection.execute(CREATE_INDEX_CHUNK_ID_SQL)
            self.connection.execute(CREATE_INDEX_DOCUMENT_ID_SQL)
            self.connection.execute(CREATE_DB_META_TABLE_SQL)
            self.connection.execute(CREATE_DOCUMENT_REGISTRY_TABLE_SQL)
            self.connection.execute(CREATE_INDEX_DOCUMENT_REGISTRY_CHECKSUM_SQL)

        # FTS5 full-text index for hybrid retrieval. Created inside its own
        # guarded block: on a minimal SQLite build without FTS5, we degrade
        # gracefully to pure-vector search rather than crashing startup.
        try:
            with self._lock, self.connection:
                self.connection.execute(CREATE_CHUNKS_FTS_SQL)
        except sqlite3.OperationalError as exc:
            self._fts_available = False
            logger.warning(
                "FTS5 unavailable (%s); hybrid search disabled, "
                "falling back to pure-vector retrieval",
                exc,
            )

        logger.info("Database schema initialized")

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a single statement within an implicit transaction."""
        with self._lock, self.connection:
            return self.connection.execute(sql, params)

    def executemany(self, sql: str, params_list: list[tuple]) -> sqlite3.Cursor:
        """Execute a statement against many parameter sets in one transaction."""
        with self._lock, self.connection:
            return self.connection.executemany(sql, params_list)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run several heterogeneous statements as a single atomic commit.

        Unlike :meth:`execute` (which commits after every call - fine
        for one-off writes, but means an N-row insert loop does N
        separate commits), this holds the lock and defers commit until
        the whole ``with`` block succeeds. Used by
        :meth:`VectorStore.insert_many` so a large ingestion batch is
        one disk sync instead of thousands.

        Example:
            with database.transaction() as conn:
                for row in rows:
                    conn.execute(sql, row)
        """
        with self._lock, self.connection:
            yield self.connection

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Run a read-only query and return all rows."""
        with self._lock:
            cursor = self.connection.execute(sql, params)
            return cursor.fetchall()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self.connection.close()
        logger.info("Database connection closed")
