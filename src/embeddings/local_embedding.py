"""Local in-process embedding provider powered by fastembed (ONNX).

Runs entirely on the machine with no external service, API key, or
billing. The chosen model is downloaded once into the local fastembed
cache on first use and reused afterwards.
"""

from __future__ import annotations

from fastembed import TextEmbedding

from src.chunking.chunker import Chunk
from src.embeddings.base_embedding import (
    BaseEmbedding,
    EmbeddedChunk,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LocalEmbedding(BaseEmbedding):

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
    ):
        self.model = model
        self.batch_size = batch_size
        self._dimensions: int | None = None
        self._model: TextEmbedding | None = None

    def _ensure_model(self) -> TextEmbedding:
        if self._model is None:
            logger.info(
                "Loading local embedding model %s (first call may download it)...",
                self.model,
            )
            self._model = TextEmbedding(
                model_name=self.model,
                batch_size=self.batch_size,
            )
        return self._model

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        model = self._ensure_model()
        vector = next(iter(model.query_embed([text])))
        self._dimensions = len(vector)
        return [float(v) for v in vector]

    def embed_documents(
        self,
        chunks: list[Chunk],
    ) -> list[EmbeddedChunk]:

        if not chunks:
            return []

        logger.info(
            "Embedding %d chunks using local %s (batch_size=%d)",
            len(chunks),
            self.model,
            self.batch_size,
        )

        model = self._ensure_model()
        embedded_chunks: list[EmbeddedChunk] = []

        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            texts = [chunk["text"] for chunk in batch]
            vectors = list(model.embed(texts))

            if vectors and self._dimensions is None:
                self._dimensions = len(vectors[0])

            for chunk, vector in zip(batch, vectors):
                embedded_chunks.append(
                    {
                        "text": chunk["text"],
                        "embedding": [float(v) for v in vector],
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