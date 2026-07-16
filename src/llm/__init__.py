"""LLM provider package.

Public API::

    from src.llm.base_llm import BaseLLM, ChatMessage, LLMError
    from src.llm.sllm import SLLMChat               # SharedLLM remote chat
    from src.llm.sharedllm_chat import SharedLLMChat  # alias
    from src.llm.ollama_chat import OllamaChat        # local Ollama chat (unused in hybrid mode)
"""

from src.llm.base_llm import BaseLLM, ChatMessage, LLMError
from src.llm.sharedllm_chat import SharedLLMChat, SLLMChat

__all__ = [
    "BaseLLM",
    "ChatMessage",
    "LLMError",
    "SLLMChat",
    "SharedLLMChat",
]
