"""High-level vector store API backed by SQLite + sqlite-vec.

This is the only module the rest of the pipeline (ingestion, query
scripts) should import. It hides all raw SQL behind a small,
well-documented interface.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import sqlite_vec

from src.embeddings.base_embedding import EmbeddedChunk
from src.utils.logger import get_logger
from src.vectordb.database import Database
from src.vectordb.schema import (
    CHUNKS_FTS_TABLE,
    CHUNKS_TABLE,
    DOCUMENT_REGISTRY_TABLE,
    VEC_TABLE,
    build_vec_table_sql,
)

logger = get_logger(__name__)

_DIMENSIONS_META_KEY = "embedding_dimensions"

# FTS5's default tokenizer (unicode61) does no stemming, so "refund" won't
# match "refunds". We split the query into terms and attach a prefix
# wildcard (`term*`) to each so inflectional variants match — the standard
# FTS5 query-shaping trick. Pure-punctuation/very-short terms are dropped.
_FTS_TERM_RE = re.compile(r"[A-Za-z0-9]+")


def _build_fts_query(query: str) -> str:
    """Shape a natural-language query into an FTS5 MATCH expression.

    Each alphanumeric term becomes a prefix query (``term*``), and the
    terms are OR-joined so a match on any term surfaces the document.
    Quoting each term also avoids FTS5 syntax errors on stray punctuation.
    """
    terms = [m.group(0) for m in _FTS_TERM_RE.finditer(query) if len(m.group(0)) >= 2]
    if not terms:
        return ""
    return " OR ".join(f'"{term}"*' for term in terms)


class SearchResult(TypedDict):
    """A single similarity search hit."""

    id: str
    chunk_id: str
    document_id: str | None
    text: str
    metadata: dict[str, Any]
    distance: float
    similarity: float


class VectorStoreError(Exception):
    """Raised for vector store configuration or usage errors."""


class VectorStore:
    """SQLite + sqlite-vec backed vector store.

    Args:
        database_path: Path to the SQLite database file.
        default_top_k: Default number of results returned by
            :meth:`similarity_search` when ``top_k`` isn't specified.
    """

    def __init__(self, database_path: str | Path, default_top_k: int = 5) -> None:
        self.database = Database(database_path)
        self.default_top_k = default_top_k
        self._dimensions: int | None = None
        self.initialize_database()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def initialize_database(self) -> None:
        """Create the base schema and recover any previously stored
        embedding dimensionality so the vec0 table can be re-created
        (or reused) consistently across process restarts.
        """
        self.database.initialize_database()

        rows = self.database.query(
            "SELECT value FROM db_meta WHERE key = ?", (_DIMENSIONS_META_KEY,)
        )
        if rows:
            self._dimensions = int(rows[0]["value"])
            self._ensure_vec_table(self._dimensions)
            logger.info(
                "Recovered existing vector table with dimensions=%d", self._dimensions
            )

    def _ensure_vec_table(self, dimensions: int) -> None:
        """Create the vec0 table for the given dimensionality if needed,
        raising if the store was already initialized with a different
        dimensionality (e.g. the embedding model changed).
        """
        if self._dimensions is not None and self._dimensions != dimensions:
            raise VectorStoreError(
                f"Embedding dimension mismatch: store was initialized with "
                f"{self._dimensions} dimensions but received {dimensions}. "
                "Use a fresh database if you've changed embedding models."
            )

        if self._dimensions is None:
            self.database.execute(build_vec_table_sql(dimensions))
            self.database.execute(
                "INSERT OR REPLACE INTO db_meta (key, value) VALUES (?, ?)",
                (_DIMENSIONS_META_KEY, str(dimensions)),
            )
            self._dimensions = dimensions
            logger.info("Created vec0 table with dimensions=%d", dimensions)

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def insert_document(self, embedded_chunk: EmbeddedChunk) -> None:
        """Insert a single embedded chunk into the store."""
        self.insert_many([embedded_chunk])

    def insert_many(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """Insert many embedded chunks in a single transaction.

        Args:
            embedded_chunks: Chunks with vectors, as produced by an
                embedding provider's ``embed_documents``.

        Returns:
            The number of rows actually inserted.
        """
        if not embedded_chunks:
            return 0

        dimensions = len(embedded_chunks[0]["embedding"])
        self._ensure_vec_table(dimensions)

        inserted = 0
        with self.database.transaction() as conn:
            for chunk in embedded_chunks:
                metadata = chunk.get("metadata", {})
                chunk_id = metadata.get("chunk_id")
                document_id = metadata.get("document_id")
                row_id = metadata.get("id") or chunk_id

                if len(chunk["embedding"]) != self._dimensions:
                    logger.error(
                        "Skipping chunk '%s': embedding dim %d != store dim %d",
                        chunk_id,
                        len(chunk["embedding"]),
                        self._dimensions,
                    )
                    continue

                cursor = conn.execute(
                    f"""
                    INSERT OR REPLACE INTO {CHUNKS_TABLE}
                        (id, chunk_id, document_id, text, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (row_id, chunk_id, document_id, chunk["text"], json.dumps(metadata)),
                )
                rowid = cursor.lastrowid

                conn.execute(
                    f"INSERT OR REPLACE INTO {VEC_TABLE} (rowid, embedding) VALUES (?, ?)",
                    (rowid, sqlite_vec.serialize_float32(chunk["embedding"])),
                )

                # Keep the FTS5 full-text index in sync so hybrid (BM25)
                # search sees the same text as the vector store.
                if self.database.fts_available:
                    conn.execute(
                        f"INSERT OR REPLACE INTO {CHUNKS_FTS_TABLE} (rowid, text) VALUES (?, ?)",
                        (rowid, chunk["text"]),
                    )
                inserted += 1

        logger.info("Inserted %d/%d chunks into vector store", inserted, len(embedded_chunks))
        return inserted

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def similarity_search(
        self, query_embedding: list[float], top_k: int | None = None
    ) -> list[SearchResult]:
        """Run a cosine-similarity nearest-neighbor search.

        Args:
            query_embedding: The embedding vector to search against.
            top_k: Number of results to return. Defaults to
                ``self.default_top_k`` (configurable via ``TOP_K``).

        Returns:
            Results ordered from most to least similar.
        """
        if self._dimensions is None:
            logger.warning("similarity_search called on an empty vector store")
            return []

        if len(query_embedding) != self._dimensions:
            raise VectorStoreError(
                f"Query embedding has {len(query_embedding)} dimensions, "
                f"expected {self._dimensions}"
            )

        k = int(top_k or self.default_top_k)

        # sqlite-vec's vec0 KNN queries require the `k` constraint to be
        # a literal integer in the WHERE clause (a bound `?` parameter
        # is not recognized as satisfying the KNN LIMIT requirement).
        # `k` is always an int here, so this is not a SQL-injection risk.
        rows = self.database.query(
            f"""
            SELECT
                c.id            AS id,
                c.chunk_id      AS chunk_id,
                c.document_id   AS document_id,
                c.text          AS text,
                c.metadata      AS metadata,
                v.distance      AS distance
            FROM {VEC_TABLE} v
            JOIN {CHUNKS_TABLE} c ON c.rowid = v.rowid
            WHERE v.embedding MATCH ? AND k = {k}
            ORDER BY v.distance
            """,
            (sqlite_vec.serialize_float32(query_embedding),),
        )

        results: list[SearchResult] = []
        for row in rows:
            # sqlite-vec's cosine distance = 1 - cosine_similarity
            similarity = 1.0 - row["distance"]
            results.append(
                {
                    "id": row["id"],
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "text": row["text"],
                    "metadata": json.loads(row["metadata"]),
                    "distance": row["distance"],
                    "similarity": similarity,
                }
            )

        logger.info("similarity_search returned %d results (top_k=%d)", len(results), k)
        return results

    def keyword_search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Run a BM25 keyword search over chunk text via the FTS5 index.

        Complements :meth:`similarity_search` (dense vector retrieval) by
        catching exact-term / keyword matches that embedding similarity can
        miss. The retriever fuses the two ranked lists via Reciprocal Rank
        Fusion.

        Args:
            query: The natural-language query, matched against chunk text.
            limit: Maximum number of hits to return.

        Returns:
            Results ordered from most to least BM25-relevant. ``similarity``
            is a normalized score in ``[0, 1]`` derived from the FTS5 rank so
            it's comparable with vector-similarity scores during fusion.
            Returns an empty list if FTS5 is unavailable or nothing matches.
        """
        if not self.database.fts_available:
            return []

        query = (query or "").strip()
        if not query:
            return []

        fts_query = _build_fts_query(query)
        if not fts_query:
            return []

        limit = max(1, int(limit))

        try:
            rows = self.database.query(
                f"""
                SELECT
                    c.id            AS id,
                    c.chunk_id      AS chunk_id,
                    c.document_id   AS document_id,
                    c.text          AS text,
                    c.metadata      AS metadata,
                    bm25({CHUNKS_FTS_TABLE}) AS rank
                FROM {CHUNKS_FTS_TABLE}
                JOIN {CHUNKS_TABLE} c ON c.rowid = {CHUNKS_FTS_TABLE}.rowid
                WHERE {CHUNKS_FTS_TABLE} MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            )
        except Exception as exc:  # noqa: BLE001 - malformed query etc. must not crash retrieval
            logger.warning("keyword_search failed for query '%s': %s", query, exc)
            return []

        results: list[SearchResult] = []
        for row in rows:
            # FTS5's bm25() returns negative values (more negative = more
            # relevant). Normalize to a [0, 1] similarity-like score so it
            # can be fused with cosine similarities by the retriever.
            rank = row["rank"] if row["rank"] is not None else 0.0
            similarity = 1.0 / (1.0 + abs(float(rank)))
            results.append(
                {
                    "id": row["id"],
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "text": row["text"],
                    "metadata": json.loads(row["metadata"]),
                    "distance": abs(float(rank)),
                    "similarity": similarity,
                }
            )

        logger.info("keyword_search returned %d results (limit=%d)", len(results), limit)
        return results

    def count(self) -> int:
        """Return the total number of chunks stored."""
        rows = self.database.query(f"SELECT COUNT(*) AS total FROM {CHUNKS_TABLE}")
        return int(rows[0]["total"]) if rows else 0

    # ------------------------------------------------------------------ #
    # Deletes
    # ------------------------------------------------------------------ #

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a given document ID.

        Returns:
            The number of chunks deleted.
        """
        rows = self.database.query(
            f"SELECT rowid FROM {CHUNKS_TABLE} WHERE document_id = ?", (document_id,)
        )
        rowids = [row["rowid"] for row in rows]

        if not rowids:
            logger.info("No chunks found for document_id='%s'", document_id)
            return 0

        placeholders = ",".join("?" * len(rowids))
        self.database.execute(
            f"DELETE FROM {CHUNKS_TABLE} WHERE rowid IN ({placeholders})", tuple(rowids)
        )
        if self._dimensions is not None:
            self.database.execute(
                f"DELETE FROM {VEC_TABLE} WHERE rowid IN ({placeholders})", tuple(rowids)
            )
        if self.database.fts_available:
            self.database.execute(
                f"DELETE FROM {CHUNKS_FTS_TABLE} WHERE rowid IN ({placeholders})", tuple(rowids)
            )

        logger.info("Deleted %d chunks for document_id='%s'", len(rowids), document_id)
        self.unregister_document(document_id)
        return len(rowids)

    def delete_all(self) -> None:
        """Delete every chunk and embedding from the store."""
        self.database.execute(f"DELETE FROM {CHUNKS_TABLE}")
        if self._dimensions is not None:
            self.database.execute(f"DELETE FROM {VEC_TABLE}")
        if self.database.fts_available:
            self.database.execute(f"DELETE FROM {CHUNKS_FTS_TABLE}")
        self.database.execute(f"DELETE FROM {DOCUMENT_REGISTRY_TABLE}")
        logger.info("Deleted all chunks from vector store")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Close the underlying database connection."""
        self.database.close()

    # ------------------------------------------------------------------ #
    # Phase 6: document registry (dedup / incremental ingestion)
    # ------------------------------------------------------------------ #
    # These methods are additive on top of the Phase 2-5 vector store -
    # they don't change how `chunks` / `chunks_vec` behave, they only
    # track which source files have already been ingested so re-running
    # the ingestion pipeline is idempotent.

    def is_document_ingested(self, checksum: str) -> str | None:
        """Return the existing ``document_id`` if this checksum was
        already ingested, or ``None`` if it's new content.

        Args:
            checksum: A content hash (e.g. SHA-256) of the source file.
        """
        rows = self.database.query(
            f"SELECT document_id FROM {DOCUMENT_REGISTRY_TABLE} WHERE checksum = ?",
            (checksum,),
        )
        return rows[0]["document_id"] if rows else None

    def register_document(
        self,
        document_id: str,
        checksum: str,
        filename: str,
        source: str | None,
        chunk_count: int,
    ) -> None:
        """Record that a document has been fully ingested.

        Called once per source file, after its chunks have been
        successfully embedded and stored, so a re-run of the ingestion
        pipeline can skip it via :meth:`is_document_ingested`.
        """
        ingested_at = datetime.now(timezone.utc).isoformat()
        self.database.execute(
            f"""
            INSERT OR REPLACE INTO {DOCUMENT_REGISTRY_TABLE}
                (document_id, checksum, filename, source, chunk_count, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (document_id, checksum, filename, source, chunk_count, ingested_at),
        )
        logger.info(
            "Registered document '%s' (id=%s, %d chunks)", filename, document_id, chunk_count
        )

    def list_ingested_documents(self) -> list[dict[str, Any]]:
        """Return metadata for every document registered so far."""
        rows = self.database.query(
            f"SELECT * FROM {DOCUMENT_REGISTRY_TABLE} ORDER BY ingested_at DESC"
        )
        return [dict(row) for row in rows]

    def unregister_document(self, document_id: str) -> None:
        """Remove a document's registry entry (e.g. after deleting its chunks)."""
        self.database.execute(
            f"DELETE FROM {DOCUMENT_REGISTRY_TABLE} WHERE document_id = ?", (document_id,)
        )
