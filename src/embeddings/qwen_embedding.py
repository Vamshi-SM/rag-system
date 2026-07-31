"""Embedding provider that talks to the team's internal SLLM API
(Qwen embedding models).

The SLLM API is assumed to expose an OpenAI-compatible
``POST {base_url}/embeddings`` endpoint:

    Request:
        {
            "model": "<model-name>",
            "input": ["text 1", "text 2", ...]
        }

    Response:
        {
            "data": [
                {"embedding": [...], "index": 0},
                {"embedding": [...], "index": 1}
            ]
        }

If your team's actual SLLM API shape differs, only ``_call_api``
needs to change - everything else (batching, retries, progress bar)
is reusable as-is.
"""

from __future__ import annotations

import time
from typing import Any

import requests
from tqdm import tqdm

from src.chunking.chunker import Chunk
from src.embeddings.base_embedding import BaseEmbedding, EmbeddedChunk, EmbeddingError
from src.rag import response
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QwenEmbedding(BaseEmbedding):
    """Embedding provider backed by the team's SLLM API.

    Args:
        api_key: SLLM API key (from ``SLLM_API_KEY``).
        base_url: Base URL of the SLLM API (from ``SLLM_BASE_URL``).
        model: Embedding model name, e.g. a Qwen embedding variant
            (from ``EMBEDDING_MODEL``).
        batch_size: Number of texts sent per API call.
        timeout: Per-request timeout, in seconds.
        max_retries: Number of retry attempts on transient failures.
        backoff_factor: Base delay (seconds) for exponential backoff
            between retries.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        batch_size: int = 8,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> None:
        if not api_key:
            logger.warning("SLLM_API_KEY is empty - requests will likely be rejected")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        self._dimensions: int | None = None
        self._session = requests.Session()
        self._session.headers.update(
            {
                "X-SharedLLM-Key": self.api_key,
                "Content-Type": "application/json",
            }
        )

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call the SLLM embeddings endpoint for a batch of texts.

        Retries with exponential backoff on network errors, timeouts,
        and 5xx/429 responses. Raises :class:`EmbeddingError` if all
        retries are exhausted.
        """
        url = f"{self.base_url}/embeddings"
        payload: dict[str, Any] = {"model": self.model, "input": texts}

        last_exception: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.post(url, json=payload, timeout=self.timeout)

                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.exceptions.HTTPError(
                        f"Retryable status {response.status_code}: {response.text[:200]}"
                    )

                response.raise_for_status()
                body = response.json()

                data = sorted(body["data"], key=lambda item: item.get("index", 0))
                embeddings = [item["embedding"] for item in data]

                if embeddings and self._dimensions is None:
                    self._dimensions = len(embeddings[0])

                return embeddings

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                KeyError,
                ValueError,
            ) as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    delay = self.backoff_factor * (2 ** (attempt - 1))
                    logger.warning(
                        "Embedding API call failed (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt,
                        self.max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Embedding API call failed after %d attempts: %s",
                        self.max_retries,
                        exc,
                    )

        raise EmbeddingError(
            f"Failed to embed batch after {self.max_retries} attempts: {last_exception}"
        )

    def embed(self, text: str) -> list[float]:
        """Embed a single string."""
        embeddings = self._call_api([text])
        return embeddings[0]

    def embed_documents(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of chunks in batches, with a progress bar.

        Failed batches are logged and skipped rather than aborting the
        entire run, so a transient issue with one batch doesn't lose
        embeddings already computed for prior batches.
        """
        embedded_chunks: list[EmbeddedChunk] = []

        batches = [
            chunks[i : i + self.batch_size] for i in range(0, len(chunks), self.batch_size)
        ]

        for batch in tqdm(batches, desc="Embedding chunks", unit="batch"):
            texts = [chunk["text"] for chunk in batch]
            try:
                vectors = self._call_api(texts)
            except EmbeddingError as exc:
                logger.error(
                    "Skipping batch of %d chunks due to embedding failure: %s",
                    len(batch),
                    exc,
                )
                continue

            for chunk, vector in zip(batch, vectors):
                embedded_chunks.append(
                    {
                        "text": chunk["text"],
                        "embedding": vector,
                        "metadata": {
                            "chunk_id": chunk["chunk_id"],
                            "document_id": chunk["document_id"],
                            **chunk["metadata"],
                        },
                    }
                )

        logger.info(
            "Generated %d/%d embeddings (dimensions=%s)",
            len(embedded_chunks),
            len(chunks),
            self._dimensions,
        )
        return embedded_chunks
