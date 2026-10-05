from __future__ import annotations

import os
from typing import Iterator

from google import genai
from google.genai import types as gtypes

from src.llm.base_llm import BaseLLM, ChatMessage, LLMError


class GeminiChat(BaseLLM):
    def __init__(
        self,
        project_id: str,
        location: str = "asia-south1",
        model: str = "gemini-2.5-flash",
        thinking_budget: int | None = None,
    ):
        self.project_id = project_id
        self.location = location
        self.model = model
        # gemini-2.5-flash "thinks" before its first output token by default
        # (seconds of invisible latency). Grounded RAG answers are short
        # extractions - thinking adds prefill time, not quality - so the
        # default budget is 0 (disabled); GEMINI_THINKING_BUDGET restores it.
        if thinking_budget is None:
            thinking_budget = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
        self._config = (
            gtypes.GenerateContentConfig(
                thinking_config=gtypes.ThinkingConfig(thinking_budget=thinking_budget)
            )
            if thinking_budget != -1  # -1 = model default (dynamic thinking)
            else None
        )
        self.client = None

    def _ensure_client(self):
        # The genai.Client resolves credentials at construction time; build
        # it lazily so the API server can boot (and serve LLM-free routes
        # like /query/latency and /health) even before GCP credentials
        # or GCP_PROJECT_ID are configured.
        if self.client is None:
            self.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
            )
        return self.client

    def generate(self, prompt: str) -> str:
        try:
            response = self._ensure_client().models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._config,
            )
            return response.text
        except Exception as e:
            raise LLMError(str(e))

    def chat(self, messages: list[ChatMessage]) -> str:
        prompt = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        return self.generate(prompt)

    def stream(self, messages: list[ChatMessage]) -> Iterator[str]:
        """Stream completion tokens chunk-by-chunk using Gemini stream API."""
        prompt = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        try:
            response = self._ensure_client().models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=self._config,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMError(str(e))