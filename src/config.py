"""Application configuration loaded from environment variables.

All configuration values used across Phases 2-5 are centralized here so
that no module hardcodes paths, credentials, or tunable parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()


def _get_int(key: str, default: int) -> int:
    value = os.getenv(key)
    try:
        return int(value) if value else default
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    value = os.getenv(key)
    try:
        return float(value) if value else default
    except ValueError:
        return default


@dataclass
class Settings:
    """Strongly-typed application settings.

    Every field maps directly to an environment variable so behavior can
    be changed without touching source code.

    Note: intentionally *not* frozen. The application itself never
    mutates ``settings`` after startup, but the test suite (Phase 10)
    needs to point individual fields (e.g. ``database_path``) at
    temporary directories for isolated, side-effect-free test runs.
    """

    # --- Paths ---
    data_directory: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIRECTORY", "data/documents"))
    )
    processed_directory: Path = field(
        default_factory=lambda: Path(os.getenv("PROCESSED_DIRECTORY", "data/processed"))
    )
    database_path: Path = field(
        default_factory=lambda: Path(os.getenv("DATABASE_PATH", "data/processed/vectors.db"))
    )
    database_url: str = field(
    default_factory=lambda: os.getenv("DATABASE_URL", "")
)
    database_backend: str = field(
    default_factory=lambda: os.getenv("DATABASE_BACKEND", "sqlite")
)

    # --- Chunking ---
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 700))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 100))

    # --- Embeddings (SharedLLM / OpenAI-compatible API) ---
    sllm_api_key: str = field(
        default_factory=lambda: os.getenv("SHAREDLLM_API_KEY", os.getenv("SLLM_API_KEY", ""))
    )
    sllm_base_url: str = field(
        default_factory=lambda: os.getenv(
            "SHAREDLLM_BASE_URL",
            os.getenv("SLLM_BASE_URL", "https://api.sharedllm.com/openai/v1"),
        )
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    embedding_model: str = field(
    default_factory=lambda: os.getenv(
        "EMBEDDING_MODEL",
        "text-embedding-004",
    )
)
    embedding_batch_size: int = field(default_factory=lambda: _get_int("EMBEDDING_BATCH_SIZE", 32))
    embedding_timeout: float = field(default_factory=lambda: _get_float("EMBEDDING_TIMEOUT", 30.0))
    embedding_max_retries: int = field(default_factory=lambda: _get_int("EMBEDDING_MAX_RETRIES", 3))
    #: Number of embedding batches sent to the embedding API concurrently
    #: during ingestion. The embedding stage is purely HTTP I/O (DB writes
    #: happen afterward in ``insert_many``), so parallelizing batches cuts
    #: ingestion wall-clock time roughly linearly up to the API's rate limit.
    #: Set to 1 to keep the original strictly-sequential behavior.
    embedding_concurrency: int = field(
        default_factory=lambda: _get_int("EMBEDDING_CONCURRENCY", 4)
    )

    # --- Vector DB / Retrieval ---
    top_k: int = field(default_factory=lambda: _get_int("TOP_K", 5))
    similarity_threshold: float = field(
        default_factory=lambda: _get_float("SIMILARITY_THRESHOLD", 0.7)
    )
    #: How many extra candidates to pull from the vector store before
    #: applying metadata/filename filters and the similarity threshold,
    #: so post-filtering doesn't starve results. See src/retrieval/retriever.py.
    retrieval_candidate_multiplier: int = field(
        default_factory=lambda: _get_int("RETRIEVAL_CANDIDATE_MULTIPLIER", 4)
    )
    # --- Google Cloud / Gemini ---
    gcp_project_id: str = field(
        default_factory=lambda: os.getenv("GCP_PROJECT_ID", "")
)

    gcp_location: str = field(
        default_factory=lambda: os.getenv("GCP_LOCATION", "asia-south1")
)

    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
)
    # --- LLM (SharedLLM chat API) ---
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "ollama/kimi-k2.7-code"))
    request_timeout: float = field(default_factory=lambda: _get_float("REQUEST_TIMEOUT", 60.0))
    llm_max_retries: int = field(default_factory=lambda: _get_int("LLM_MAX_RETRIES", 3))
    #: Base URL for the chat LLM.  Defaults to the Ollama-compatible
    #: endpoint (https://api.sharedllm.com/ollama).  Embeddings use
    #: ``sllm_base_url`` which points at the OpenAI-compatible endpoint.
    #: Override via ``SHAREDLLM_CHAT_BASE_URL``.
    chat_base_url: str = field(
        default_factory=lambda: os.getenv(
            "SHAREDLLM_CHAT_BASE_URL",
            os.getenv("SLLM_BASE_URL", "https://api.sharedllm.com/ollama"),
        )
    )

    # --- API security ---
    #: POST /ingest accepts a folder path from the caller. To prevent
    #: path traversal (a caller pointing the API at /etc, ~/.ssh, etc.),
    #: any requested folder must resolve to a path inside this root.
    #: Defaults to the parent of DATA_DIRECTORY (typically ``data/``).
    allowed_ingest_root: Path = field(
        default_factory=lambda: Path(
            os.getenv("ALLOWED_INGEST_ROOT", os.getenv("DATA_DIRECTORY", "data/documents"))
        ).parent
    )
    #: Comma-separated list of allowed CORS origins for the API.
    #: Empty by default (no cross-origin access); set to "*" to allow all
    #: (development only) or a comma-separated list of specific origins.
    allowed_cors_origins: list[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv("ALLOWED_CORS_ORIGINS", "").split(",")
            if origin.strip()
        ]
    )

    # --- Logging ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_directory: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIRECTORY", "logs")))

    def ensure_directories(self) -> None:
        """Create all directories this configuration depends on."""
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.processed_directory.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
