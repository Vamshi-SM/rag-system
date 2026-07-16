"""Phase 8 - Prompt construction for the RAG pipeline.

Builds the exact context+question prompt sent to the LLM, always
instructing it to answer only from retrieved context so the pipeline
never hallucinates beyond what was actually retrieved.
"""

from __future__ import annotations

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


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a numbered, source-attributed context block.

    Each chunk is labeled with its filename and page (when known) so
    the LLM's answer can be traced back to a source, and so
    ``response.py`` can independently reconstruct the sources list.
    """
    if not chunks:
        return "(no relevant context was found)"

    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        filename = chunk["metadata"].get("filename", "unknown source")
        page = chunk["metadata"].get("page")
        location = f"{filename}, page {page}" if page is not None else filename
        blocks.append(f"[{index}] ({location})\n{chunk['text'].strip()}")

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
