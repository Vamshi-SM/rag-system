from __future__ import annotations

from typing import Iterator
from google import genai

from src.llm.base_llm import BaseLLM, ChatMessage, LLMError


class GeminiChat(BaseLLM):
    def __init__(
        self,
        project_id: str,
        location: str = "asia-south1",
        model: str = "gemini-2.5-flash",
    ):
        self.project_id = project_id
        self.location = location
        self.model = model
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
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMError(str(e))