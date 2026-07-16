"""Abstract interface for chat/completion LLM providers.

Mirrors the shape of ``src/embeddings/base_embedding.py`` so the two
provider layers (embeddings, chat) follow the same pattern: an abstract
base class plus one concrete SLLM implementation, swappable without
touching the rest of the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, TypedDict


class ChatMessage(TypedDict):
    """A single chat message in OpenAI-style role/content format."""

    role: str  # "system" | "user" | "assistant"
    content: str


class LLMError(Exception):
    """Raised when the LLM backend fails after exhausting retries."""


class BaseLLM(ABC):
    """Abstract base class all chat/completion LLM providers must implement."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a completion for a single prompt string.

        A thin convenience over :meth:`chat` for callers that just have
        one already-assembled prompt (e.g. the RAG pipeline's final
        context+question prompt) rather than a full message history.
        """
        raise NotImplementedError

    @abstractmethod
    def chat(self, messages: list[ChatMessage]) -> str:
        """Generate a completion for a full multi-turn message history."""
        raise NotImplementedError

    def stream(self, messages: list[ChatMessage]) -> Iterator[str]:
        """Stream a completion token-by-token (or chunk-by-chunk).

        Optional: default implementation falls back to yielding the
        full response from :meth:`chat` as a single chunk, so callers
        can always call ``stream()`` even against a provider that
        hasn't implemented true streaming.
        """
        yield self.chat(messages)
