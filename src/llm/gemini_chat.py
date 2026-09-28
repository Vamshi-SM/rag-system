from __future__ import annotations

import os
import time
from typing import Iterator

from google import genai

from src.llm.base_llm import BaseLLM, ChatMessage, LLMError

_TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")


class GeminiChat(BaseLLM):
    def __init__(self, project_id: str, location: str = "asia-south1", model: str | None = None):
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )
        self.model = model or "gemini-2.5-flash"

    def generate(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                message = str(e)
                transient = any(marker in message for marker in _TRANSIENT_MARKERS)
                if transient and attempt < 2:
                    delay = 2.0 * (attempt + 1)
                    time.sleep(delay)
                    continue
                raise LLMError(message) from e
        raise LLMError("LLM generation failed after retries")

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
            response = self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMError(str(e))