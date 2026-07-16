"""SQL schema definitions for the SQLite + sqlite-vec vector store.

Two tables work together:

* ``chunks`` - a regular table holding the text, JSON metadata, and IDs.
* ``chunks_vec`` - a ``vec0`` virtual table (from the sqlite-vec
  extension) holding only the embedding, keyed by the same ``rowid`` as
  ``chunks``. The virtual table's dimensionality is only known once the
  first embedding arrives, so it is created lazily by
  :mod:`src.vectordb.database` rather than at import time.
"""

from __future__ import annotations

CHUNKS_TABLE = "chunks"
VEC_TABLE = "chunks_vec"

CREATE_CHUNKS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {CHUNKS_TABLE} (
    rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
    id          TEXT UNIQUE NOT NULL,
    chunk_id    TEXT NOT NULL,
    document_id TEXT,
    text        TEXT NOT NULL,
    metadata    TEXT NOT NULL
);
"""

CREATE_INDEX_CHUNK_ID_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_{CHUNKS_TABLE}_chunk_id ON {CHUNKS_TABLE}(chunk_id);
"""

CREATE_INDEX_DOCUMENT_ID_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_{CHUNKS_TABLE}_document_id ON {CHUNKS_TABLE}(document_id);
"""

# Small metadata table tracking the embedding dimensionality that the
# vec0 table was created with, so we can detect mismatches on later
# inserts (e.g. someone switching embedding models mid-project).
CREATE_DB_META_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS db_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def build_vec_table_sql(dimensions: int, distance_metric: str = "cosine") -> str:
    """Build the CREATE VIRTUAL TABLE statement for the vec0 table.

    Args:
        dimensions: Embedding vector dimensionality. Never hardcoded -
            determined dynamically from the first embedding produced by
            whichever Qwen embedding model is configured.
        distance_metric: Distance metric sqlite-vec should use internally.
    """
    return (
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {VEC_TABLE} USING vec0("
        f"embedding float[{dimensions}] distance_metric={distance_metric}"
        f");"
    )


# --------------------------------------------------------------------- #
# Phase 6: document registry (dedup / incremental ingestion)
# --------------------------------------------------------------------- #
# A lightweight append-only table used purely to detect whether a given
# file's *content* has already been ingested, so re-running ingest.py
# never inserts duplicate chunks for an unchanged file. This table is
# additive - it doesn't change how `chunks` / `chunks_vec` work at all.

DOCUMENT_REGISTRY_TABLE = "document_registry"

CREATE_DOCUMENT_REGISTRY_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {DOCUMENT_REGISTRY_TABLE} (
    document_id  TEXT PRIMARY KEY,
    checksum     TEXT UNIQUE NOT NULL,
    filename     TEXT NOT NULL,
    source       TEXT,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    ingested_at  TEXT NOT NULL
);
"""

CREATE_INDEX_DOCUMENT_REGISTRY_CHECKSUM_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_{DOCUMENT_REGISTRY_TABLE}_checksum
ON {DOCUMENT_REGISTRY_TABLE}(checksum);
"""
