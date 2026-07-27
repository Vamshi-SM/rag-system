from __future__ import annotations

import time
from google import genai
from google.genai.types import EmbedContentConfig
from tqdm import tqdm

from src.chunking.chunker import Chunk
from src.embeddings.base_embedding import (
    BaseEmbedding,
    EmbeddedChunk,
)
from src.utils.logger import get_logger
from google.genai.errors import ClientError
logger = get_logger(__name__)


class GoogleEmbedding(BaseEmbedding):

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model: str = "text-embedding-004",
        batch_size: int = 8,
    ):
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )

        self.model = model
        self.batch_size = batch_size
        self._dimensions = None

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def _embed_with_retry(
        self, contents: str | list[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ):
        """Execute embed_content with exponential backoff on 429 / Rate Limits."""
        max_retries = 5
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                return self.client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=EmbedContentConfig(
                        task_type=task_type,
                    ),
                )
            except ClientError as e:
                err_msg = str(e)
                if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Rate limit / 429 hit. Retrying batch in %.1fs (attempt %d/%d)...",
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(delay)
                else:
                    raise

    def embed(self, text: str) -> list[float]:
        response = self._embed_with_retry(contents=text, task_type="RETRIEVAL_DOCUMENT")

        vector = response.embeddings[0].values

        if self._dimensions is None:
            self._dimensions = len(vector)

        return vector

    def embed_documents(
        self,
        chunks: list[Chunk],
    ) -> list[EmbeddedChunk]:

        if not chunks:
            return []

        embedded_chunks: list[EmbeddedChunk] = []

        logger.info(
            "Embedding %d chunks using Google %s (batch_size=%d)",
            len(chunks),
            self.model,
            self.batch_size,
        )

        # Build clean batches respecting both batch_size and max payload characters
        MAX_CHARS = 20000
        batches: list[list[Chunk]] = []
        current_batch: list[Chunk] = []
        current_chars = 0

        for chunk in chunks:
            text = chunk["text"]
            
            # Finalize batch if adding chunk exceeds batch_size OR MAX_CHARS limit
            if current_batch and (
                len(current_batch) >= self.batch_size
                or current_chars + len(text) > MAX_CHARS
            ):
                batches.append(current_batch)
                current_batch = []
                current_chars = 0

            current_batch.append(chunk)
            current_chars += len(text)

        if current_batch:
            batches.append(current_batch)

        logger.info("Grouped %d chunks into %d batches", len(chunks), len(batches))

        # Process each batch
        for batch in tqdm(
            batches,
            desc="Embedding chunks",
            unit="batch",
        ):
            texts = [chunk["text"] for chunk in batch]

            response = self._embed_with_retry(contents=texts, task_type="RETRIEVAL_DOCUMENT")

            vectors = [emb.values for emb in response.embeddings]

            if vectors and self._dimensions is None:
                self._dimensions = len(vectors[0])

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
            "Generated %d embeddings (dimensions=%s)",
            len(embedded_chunks),
            self._dimensions,
        )

        return embedded_chunks