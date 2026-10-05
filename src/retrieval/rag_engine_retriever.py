"""RAG Engine (Vertex AI) retriever.

Mirrors the local ``Retriever.retrieve_with_scores`` interface
(``{text, score, metadata}`` results) so benchmark code can treat the
two engines uniformly.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

warnings.filterwarnings("ignore")

import agentplatform
from agentplatform._genai.types.common import RagQuery
from google.genai import types as gtypes

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RagEngineRetrieverError(Exception):
    pass


class RagEngineRetriever:
    """Retrieve contexts from a Vertex AI RAG Engine corpus.

    The RAG Engine embeds the query with the corpus's own embedding model
    (text-embedding-005 here) and searches its managed vector DB server-
    side; we only pay for the retrieve call itself.
    """

    def __init__(
        self,
        project_id: str | None = None,
        location: str | None = None,
        corpus_name: str | None = None,
        default_top_k: int | None = None,
    ):
        self.project_id = project_id or settings.gcp_project_id
        self.location = location or settings.gcp_location
        self.corpus_name = corpus_name or settings.rag_corpus_name
        self.default_top_k = default_top_k or settings.top_k
        if not self.corpus_name:
            raise RagEngineRetrieverError(
                "RAG_CORPUS_NAME is not set - run scripts/setup_rag_engine.py first"
            )
        self.client = agentplatform.Client(
            project=self.project_id, location=self.location
        )

    def retrieve_with_scores(
        self,
        question: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-k chunks with scores in the local retriever's shape."""
        top_k = top_k or self.default_top_k
        query = RagQuery(text=question, similarity_top_k=top_k)
        t0 = time.perf_counter()
        try:
            response = self.client.rag.retrieve_contexts(
                vertex_rag_store=gtypes.VertexRagStore(
                    rag_corpora=[self.corpus_name],
                ),
                query=query,
            )
        except Exception as exc:  # noqa: BLE001
            raise RagEngineRetrieverError(
                f"RAG Engine retrieval failed: {exc}"
            ) from exc
        elapsed = time.perf_counter() - t0
        logger.info(
            "RAG Engine retrieved %d contexts in %.2fs",
            len(self._contexts(response)), elapsed,
        )

        results: list[dict[str, Any]] = []
        for ctx in self._contexts(response):
            score = ctx.score
            if score is None and ctx.distance is not None:
                score = 1.0 - ctx.distance
            chunk = ctx.chunk
            metadata: dict[str, Any] = {
                "source": ctx.source_uri,
                "filename": (
                    ctx.source_display_name.rsplit("/", 1)[-1]
                    if ctx.source_display_name
                    else None
                ),
                "chunk_id": chunk.chunk_id if chunk else None,
                "document_id": chunk.file_id if chunk else None,
            }
            if similarity_threshold is not None and score is not None:
                if score < similarity_threshold:
                    continue
            results.append({"text": ctx.text or "", "score": score or 0.0, "metadata": metadata})
        return results

    @staticmethod
    def _contexts(response) -> list:
        # agentplatform responses are pydantic models; the contexts may
        # surface at response.contexts.contexts or via a raw payload.
        ctx_obj = getattr(response, "contexts", None)
        if ctx_obj is None:
            return []
        return list(getattr(ctx_obj, "contexts", None) or [])