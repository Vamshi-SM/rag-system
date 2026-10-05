"""Phase 8 - Prompt construction for the RAG pipeline.

Builds the exact context+question prompt sent to the LLM, always
instructing it to answer only from retrieved context so the pipeline
never hallucinates beyond what was actually retrieved.
"""

from __future__ import annotations

from src.config import settings
from src.retrieval.retriever import RetrievedChunk

SYSTEM_INSTRUCTIONS = (
    "You are a helpful AI assistant.\n\n"
    "Answer ONLY using the provided context.\n\n"
    "If the answer cannot be found in the context, clearly state that "
    "the information is not available."
)

_PROMPT_TEMPLATE = """{instructions}

Context

-------------------

{retrieved_chunks}

-------------------

Question

{user_question}"""


def _trim(text: str, limit: int) -> str:
    """Trim text to ~limit chars, preferring a sentence boundary."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    dot = cut.rfind(". ")
    if dot > limit * 0.5:
        return cut[: dot + 1]
    return cut


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a numbered, source-attributed context block.

    Each chunk is labeled with its filename and page (when known) so
    the LLM's answer can be traced back to a source, and so
    ``response.py`` can independently reconstruct the sources list.

    Token-budget aware: chunks are trimmed to
    ``prompt_max_chunk_chars`` and the whole context stops at
    ``prompt_max_context_chars`` — Gemini prefill time scales with
    input tokens, and 5 full 700-token chunks measurably add seconds.
    The highest-ranked chunk keeps the most text; supporting chunks
    yield first.
    """
    if not chunks:
        return "(no relevant context was found)"

    max_chunk = settings.prompt_max_chunk_chars
    max_total = settings.prompt_max_context_chars

    blocks: list[str] = []
    total = 0
    for index, chunk in enumerate(chunks, start=1):
        text = _trim(chunk["text"].strip(), max_chunk)
        headroom = max_total - total
        if headroom <= 0 and blocks:
            break
        if len(text) > headroom:
            if headroom < 400:
                break
            text = _trim(text, headroom)
        filename = chunk["metadata"].get("filename", "unknown source")
        page = chunk["metadata"].get("page")
        location = f"{filename}, page {page}" if page is not None else filename
        blocks.append(f"[{index}] ({location})\n{text}")
        total += len(text)

    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Build the final prompt string sent to the LLM.

    Args:
        question: The user's natural-language question.
        chunks: Retrieved chunks, as returned by
            ``Retriever.retrieve`` / ``retrieve_with_scores``.

    Returns:
        A fully-formatted prompt string following the required
        "Context / Question" template.
    """
    return _PROMPT_TEMPLATE.format(
        instructions=SYSTEM_INSTRUCTIONS,
        retrieved_chunks=format_context(chunks),
        user_question=question.strip(),
    )
