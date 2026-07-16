"""Canonical import shim for the SharedLLM chat provider.

The concrete implementation lives in ``src/llm/sharedllm_chat.py``.  This
module re-exports the public symbols so that every other module in the
project can use the stable path::

    from src.llm.sllm import SLLMChat

without knowing where the implementation file is named.  This mirrors the
pattern of having a clean public API surface regardless of internal file
layout.
"""

from src.llm.sharedllm_chat import SharedLLMChat, SLLMChat

__all__ = ["SLLMChat", "SharedLLMChat"]
