from __future__ import annotations

from typing import Iterator
from google import genai

from src.llm.base_llm import BaseLLM, ChatMessage, LLMError


class GeminiChat(BaseLLM):
    def __init__(self, project_id: str, location: str = "asia-south1"):
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )
        self.model = "gemini-2.5-flash"

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
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
            response = self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMError(str(e))