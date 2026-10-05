"""Local cross-encoder reranker.

Rescores fused retrieval candidates with a cross-encoder (query, chunk)
pairwise relevance model - a quality lever the managed RAG Engine does not
offer. Runs fully local on CPU via ONNX (fastembed TextCrossEncoder);
no API calls, no per-query cost.
"""

from __future__ import annotations

from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Rerank retrieval candidates with a local cross-encoder.

    Args:
        model_name: fastembed-supported cross-encoder model.
        top_n: how many fused candidates to rescore before returning
            the best ``top_k`` (reranking the full 50k corpus is pointless;
            candidates come pre-filtered by hybrid retrieval).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        top_n: int = 20,
        max_chars: int = 1200,
    ):
        self.model_name = model_name
        self.top_n = max(1, int(top_n))
        # Cross-encoders score (query, chunk) pairs; scoring the full
        # 700-token chunk makes CPU cost explode (~100ms/pair). The head
        # of the chunk carries the topic; the model also truncates to its
        # own context window anyway.
        self.max_chars = max(200, int(max_chars))
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            logger.info("Loading cross-encoder reranker: %s", self.model_name)
            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def rerank(
        self,
        question: str,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return the top_k most relevant results, rescored by the model.

        Best-effort contract, consistent with keyword search: any failure
        logs a warning and returns the input order unchanged rather than
        breaking retrieval. Results keep their shape; ``score`` becomes
        the cross-encoder relevance (monotonic with the returned order).
        """
        if not results:
            return []
        candidates = results[: self.top_n]
        try:
            model = self._ensure_model()
            scored = list(
                model.rerank(question, [r["text"][: self.max_chars] for r in candidates])
            )
        except Exception as exc:  # noqa: BLE001 - reranking must never break retrieval
            logger.warning("Reranking failed, keeping fusion order: %s", exc)
            return results[:top_k]

        for result, score in zip(candidates, scored):
            result["score"] = float(score)
            result["rerank_score"] = float(score)
        candidates.sort(key=lambda r: r["score"], reverse=True)
        logger.info(
            "Reranked %d candidates -> top %d", len(candidates), min(top_k, len(candidates))
        )
        return candidates[:top_k]