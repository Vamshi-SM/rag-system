"""
Chat provider backed by a local Ollama server.

Uses the Ollama chat API:

POST http://localhost:11434/api/chat

Request:

{
    "model": "qwen3:4b",
    "messages": [
        {
            "role": "user",
            "content": "Hello"
        }
    ],
    "stream": false
}

Response:

{
    "model": "...",
    "created_at": "...",
    "message": {
        "role": "assistant",
        "content": "Hello!"
    },
    "done": true
}
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterator

import requests

from src.llm.base_llm import (
    BaseLLM,
    ChatMessage,
    LLMError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaChat(BaseLLM):
    """
    Chat provider using a local Ollama server.

    Compatible with the existing RAG pipeline.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:4b",
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        temperature: float = 0.2,
    ) -> None:

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.temperature = temperature

        self._session = requests.Session()

        self._session.headers.update(
            {
                "Content-Type": "application/json",
            }
        )

    def _call_api(
        self,
        messages: list[ChatMessage],
    ) -> str:
        """
        Call the Ollama chat endpoint with retries.
        """

        url = f"{self.base_url}/api/chat"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        last_exception: Exception | None = None

        for attempt in range(
            1,
            self.max_retries + 1,
        ):

            try:

                response = self._session.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )

                if (
                    response.status_code == 429
                    or response.status_code >= 500
                ):
                    raise requests.exceptions.HTTPError(
                        f"Retryable status "
                        f"{response.status_code}: "
                        f"{response.text[:200]}"
                    )

                response.raise_for_status()

                body = response.json()

                if "message" not in body:
                    raise LLMError(
                        "Ollama response does not contain "
                        "'message'"
                    )

                return body["message"]["content"]
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                requests.exceptions.RequestException,
                ValueError,
                KeyError,
                LLMError,
            ) as exc:

                last_exception = exc

                if attempt < self.max_retries:

                    delay = self.backoff_factor * (
                        2 ** (attempt - 1)
                    )

                    logger.warning(
                        "Chat API call failed "
                        "(attempt %d/%d): %s. "
                        "Retrying in %.1fs",
                        attempt,
                        self.max_retries,
                        exc,
                        delay,
                    )

                    time.sleep(delay)

                else:

                    logger.error(
                        "Chat API failed after %d attempts: %s",
                        self.max_retries,
                        exc,
                    )

        raise LLMError(
            f"Failed to get chat completion after "
            f"{self.max_retries} attempts: "
            f"{last_exception}"
        )

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate a response from a single prompt.
        """

        return self.chat(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )

    def chat(
        self,
        messages: list[ChatMessage],
    ) -> str:
        """
        Generate a response from a conversation.
        """

        return self._call_api(messages)
    
def stream(
        self,
        messages: list[ChatMessage],
    ) -> Iterator[str]:
        """
        Stream a response from Ollama.

        Falls back to a normal (non-streaming) response if
        streaming fails.
        """

        url = f"{self.base_url}/api/chat"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
            },
        }

        try:

            with self._session.post(
                url,
                json=payload,
                timeout=self.timeout,
                stream=True,
            ) as response:

                response.raise_for_status()

                for line in response.iter_lines(decode_unicode=True):

                    if not line:
                        continue

                    try:
                        event = json.loads(line)

                        if event.get("done"):
                            break

                        message = event.get("message")

                        if not message:
                            continue

                        content = message.get("content")

                        if content:
                            yield content

                    except (
                        json.JSONDecodeError,
                        KeyError,
                        TypeError,
                    ):
                        continue

        except (
            requests.exceptions.RequestException,
            LLMError,
        ) as exc:

            logger.warning(
                "Streaming failed (%s). "
                "Falling back to non-streaming response.",
                exc,
            )

            yield self.chat(messages)