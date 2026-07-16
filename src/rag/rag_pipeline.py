"""Phase 8 - End-to-end RAG pipeline orchestrator.

Flow: question -> embed -> retrieve top chunks -> build prompt ->
send to LLM -> return a structured, source-attributed answer.

This module only composes the Phase 7 ``Retriever`` and the Phase 8
``BaseLLM``/prompt/response modules - no retrieval or generation logic
lives here directly, which keeps it easy to test and swap pieces out.
"""

from __future__ import annotations

import time

from src.llm.base_llm import BaseLLM, LLMError
from src.rag.prompt import build_prompt
from src.rag.response import NO_CONTEXT_ANSWER, RAGResponse, build_sources
from src.retrieval.retriever import Retriever, RetrievalError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """Orchestrates retrieval + prompt construction + LLM generation.

    Args:
        retriever: A configured :class:`~src.retrieval.retriever.Retriever`.
        llm: A configured :class:`~src.llm.base_llm.BaseLLM` implementation.
        top_k: Default number of chunks to retrieve per question.
        similarity_threshold: Default minimum similarity score to keep
            a retrieved chunk.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        metadata_filter: dict | None = None,
        filename: str | None = None,
    ) -> RAGResponse:
        """Answer a natural-language question using retrieved context.

        This method is designed to never raise: every failure mode
        (empty question, retrieval failure, LLM failure) is caught and
        turned into a ``RAGResponse`` with a safe fallback ``answer``
        and a populated ``error`` field, so a caller (e.g. an
        interactive CLI) can always display *something* useful.
        """
        start_time = time.time()

        question = (question or "").strip()
        if not question:
            return RAGResponse(
                answer="Please ask a question.",
                used_llm=False,
                elapsed_seconds=time.time() - start_time,
                error="empty_question",
            )

        try:
            chunks = self.retriever.retrieve_with_scores(
                question,
                top_k=top_k or self.top_k,
                similarity_threshold=similarity_threshold or self.similarity_threshold,
                metadata_filter=metadata_filter,
                filename=filename,
            )
        except RetrievalError as exc:
            logger.error("RAG pipeline: retrieval failed for question '%s': %s", question, exc)
            return RAGResponse(
                answer=(
                    "I wasn't able to search the document store right now "
                    "(a retrieval error occurred). Please try again shortly."
                ),
                used_llm=False,
                elapsed_seconds=time.time() - start_time,
                error=f"retrieval_error: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - last-resort safety net; never crash the pipeline
            logger.exception(
                "RAG pipeline: unexpected retrieval failure for question '%s': %s", question, exc
            )
            return RAGResponse(
                answer=(
                    "Something went wrong while searching the document store. "
                    "Please try again shortly."
                ),
                used_llm=False,
                elapsed_seconds=time.time() - start_time,
                error=f"unexpected_retrieval_error: {exc}",
            )

        if not chunks:
            logger.info("RAG pipeline: no chunks retrieved for question '%s'", question)
            return RAGResponse(
                answer=NO_CONTEXT_ANSWER,
                sources=[],
                chunks_used=[],
                used_llm=False,
                elapsed_seconds=time.time() - start_time,
            )

        prompt = build_prompt(question, chunks)

        try:
            answer_text = self.llm.generate(prompt)
        except LLMError as exc:
            logger.error("RAG pipeline: LLM generation failed for question '%s': %s", question, exc)
            return RAGResponse(
                answer=(
                    "I retrieved relevant context but couldn't generate an answer "
                    "right now (the language model is unavailable). Please try again shortly."
                ),
                sources=build_sources(chunks),
                chunks_used=chunks,
                used_llm=False,
                elapsed_seconds=time.time() - start_time,
                error=f"llm_error: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - last-resort safety net; never crash the pipeline
            logger.exception(
                "RAG pipeline: unexpected LLM failure for question '%s': %s", question, exc
            )
            return RAGResponse(
                answer=(
                    "I retrieved relevant context but something went wrong while "
                    "generating an answer. Please try again shortly."
                ),
                sources=build_sources(chunks),
                chunks_used=chunks,
                used_llm=False,
                elapsed_seconds=time.time() - start_time,
                error=f"unexpected_llm_error: {exc}",
            )

        elapsed = time.time() - start_time
        logger.info("RAG pipeline answered question in %.2fs (%d chunks used)", elapsed, len(chunks))

        return RAGResponse(
            answer=answer_text.strip(),
            sources=build_sources(chunks),
            chunks_used=chunks,
            used_llm=True,
            elapsed_seconds=elapsed,
        )
