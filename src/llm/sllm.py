"""Compatibility wrapper."""

from src.llm.gemini_chat import GeminiChat

SLLMChat = GeminiChat
SharedLLMChat = GeminiChat

__all__ = [
    "SLLMChat",
    "SharedLLMChat",
]