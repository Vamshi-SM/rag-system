from __future__ import annotations

from google import genai
from google.genai.types import EmbedContentConfig
from tqdm import tqdm

from src.chunking.chunker import Chunk
from src.embeddings.base_embedding import (
    BaseEmbedding,
    EmbeddedChunk,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GoogleEmbedding(BaseEmbedding):

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model: str = "text-embedding-004",
        batch_size: int = 32,
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

    def embed(self, text: str) -> list[float]:

        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
            ),
        )

        vector = response.embeddings[0].values

        if self._dimensions is None:
            self._dimensions = len(vector)

        return vector

    def embed_documents(
        self,
        chunks: list[Chunk],
    ) -> list[EmbeddedChunk]:

        embedded_chunks: list[EmbeddedChunk] = []

        logger.info(
            "Embedding %d chunks using Google %s",
            len(chunks),
            self.model,
        )

        batches = [
            chunks[i:i+self.batch_size]
            for i in range(0, len(chunks), self.batch_size)
        ]

        for batch in tqdm(
            batches,
            desc="Embedding chunks",
            unit="batch",
        ):

            texts = [
                chunk["text"]
                for chunk in batch
            ]

            response = self.client.models.embed_content(
                model=self.model,
                contents=texts,
                config=EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )

            vectors = [
                emb.values
                for emb in response.embeddings
            ]

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