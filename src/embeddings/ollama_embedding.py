"""
Embedding provider backed by Ollama.

Uses the Ollama embedding API:

POST http://localhost:11434/api/embed

Request:
{
    "model": "nomic-embed-text",
    "input": [
        "text 1",
        "text 2"
    ]
}

Response:
{
    "model": "...",
    "embeddings": [
        [...],
        [...]
    ]
}
"""

from __future__ import annotations

import time
from typing import Any

import requests
from tqdm import tqdm

from src.chunking.chunker import Chunk
from src.embeddings.base_embedding import (
    BaseEmbedding,
    EmbeddedChunk,
    EmbeddingError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaEmbedding(BaseEmbedding):
    """
    Embedding provider using a local Ollama server.

    Compatible with the existing ingestion pipeline.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        batch_size: int = 32,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> None:

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
                "Content-Type": "application/json",
            }
        )

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def _call_api(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Call Ollama's embedding endpoint.

        Automatically retries transient failures.
        """

        url = f"{self.base_url}/api/embed"

        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }

        last_exception: Exception | None = None

        for attempt in range(1, self.max_retries + 1):

            try:

                response = self._session.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.exceptions.HTTPError(
                        f"Retryable status {response.status_code}: "
                        f"{response.text[:200]}"
                    )

                response.raise_for_status()

                body = response.json()

                if "embeddings" not in body:
                    raise EmbeddingError(
                        "Ollama response does not contain 'embeddings'"
                    )

                embeddings = body["embeddings"]

                if (
                    embeddings
                    and self._dimensions is None
                ):
                    self._dimensions = len(embeddings[0])

                return embeddings

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                requests.exceptions.RequestException,
                ValueError,
                KeyError,
                EmbeddingError,
            ) as exc:

                last_exception = exc

                if attempt < self.max_retries:

                    delay = self.backoff_factor * (2 ** (attempt - 1))

                    logger.warning(
                        "Embedding API call failed "
                        "(attempt %d/%d): %s. Retrying in %.1fs",
                        attempt,
                        self.max_retries,
                        exc,
                        delay,
                    )

                    time.sleep(delay)

                else:

                    logger.error(
                        "Embedding API failed after %d attempts: %s",
                        self.max_retries,
                        exc,
                    )

        raise EmbeddingError(
            f"Failed to generate embeddings after "
            f"{self.max_retries} attempts: {last_exception}"
        )

    def embed(
        self,
        text: str,
    ) -> list[float]:
        """
        Embed a single string.
        """

        embeddings = self._call_api([text])
        return embeddings[0]

    def embed_documents(
        self,
        chunks: list[Chunk],
    ) -> list[EmbeddedChunk]:
        """
        Embed all chunks in batches.

        Failed batches are skipped so the ingestion
        process can continue.
        """

        embedded_chunks: list[EmbeddedChunk] = []

        batches = [
            chunks[i : i + self.batch_size]
            for i in range(0, len(chunks), self.batch_size)
        ]

        logger.info(
            "Embedding %d chunks using model '%s'",
            len(chunks),
            self.model,
        )

        for batch in tqdm(
            batches,
            desc="Embedding chunks",
            unit="batch",
        ):

            texts = [
                chunk["text"]
                for chunk in batch
            ]

            try:

                vectors = self._call_api(texts)

            except EmbeddingError as exc:

                logger.error(
                    "Skipping batch of %d chunks: %s",
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