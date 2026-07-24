import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from contextlib import contextmanager

class PostgresDatabase:
    def __init__(self, database_url: str):
        # row_factory=dict_row ensures rows behave like dictionaries
        self.connection = psycopg.connect(
            database_url,
            row_factory=dict_row,
        )
        self.connection.autocommit = True
        register_vector(self.connection)

    def initialize_database(self) -> None:
        """Create the necessary extensions, tables, and indexes."""
        with self.connection.cursor() as cur:
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

    def execute(self, query: str, params: tuple | None = None) -> None:
        """Execute a write query outside of a manual transaction block."""
        with self.connection.cursor() as cur:
            cur.execute(query, params)

    def query(self, query: str, params: tuple | None = None) -> list[dict]:
        """Execute a read query and return the results as a list of dicts."""
        with self.connection.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    @contextmanager
    def transaction(self):
        """Context manager for executing multiple writes in a single transaction."""
        original_autocommit = self.connection.autocommit
        self.connection.autocommit = False
        try:
            with self.connection.transaction():
                # Yield the connection so the caller can use its cursors directly
                yield self.connection
        finally:
            self.connection.autocommit = original_autocommit

    def close(self) -> None:
        """Close the database connection."""
        if self.connection and not self.connection.closed:
            self.connection.close()