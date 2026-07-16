"""Chat/completion provider that talks to the team's internal SLLM API.

Assumes an OpenAI-compatible ``POST {base_url}/chat/completions``
endpoint:

    Request:
        {
            "model": "<model-name>",
            "messages": [{"role": "user", "content": "..."}],
            "stream": false
        }

    Response:
        {
            "choices": [
                {"message": {"role": "assistant", "content": "..."}}
            ]
        }

If your team's actual SLLM chat API shape differs, only
``_call_api``/``_call_api_streaming`` need to change - retry, timeout,
and error-handling logic is reusable as-is. This mirrors the structure
of ``src/embeddings/qwen_embedding.py`` intentionally, so the two
provider modules stay easy to read side by side.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterator

import requests

from src.llm.base_llm import BaseLLM, ChatMessage, LLMError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SLLMChat(BaseLLM):
    """Chat/completion provider backed by the team's SLLM API.

    Args:
        api_key: SLLM API key (from ``SLLM_API_KEY``).
        base_url: Base URL of the SLLM API (from ``SLLM_BASE_URL``).
        model: Chat model name (from ``CHAT_MODEL``).
        timeout: Per-request timeout, in seconds.
        max_retries: Number of retry attempts on transient failures.
        backoff_factor: Base delay (seconds) for exponential backoff
            between retries.
        temperature: Sampling temperature passed to the API.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        temperature: float = 0.2,
    ) -> None:
        if not api_key:
            logger.warning("SLLM_API_KEY is empty - chat requests will likely be rejected")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.temperature = temperature

        self._session = requests.Session()
        self._session.headers.update(
            {
                "X-SharedLLM-Key": self.api_key,
                "Content-Type": "application/json",
            }
        )

    def _call_api(self, messages: list[ChatMessage]) -> str:
        """Call the SLLM chat completions endpoint with retries/backoff."""
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }

        last_exception: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.post(url, json=payload, timeout=self.timeout)

                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.exceptions.HTTPError(
                        f"Retryable status {response.status_code}: {response.text[:200]}"
                    )

                response.raise_for_status()
                body = response.json()
                return body["choices"][0]["message"]["content"]

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                KeyError,
                IndexError,
                ValueError,
            ) as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    delay = self.backoff_factor * (2 ** (attempt - 1))
                    logger.warning(
                        "Chat API call failed (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt,
                        self.max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Chat API call failed after %d attempts: %s", self.max_retries, exc
                    )

        raise LLMError(f"Failed to get chat completion after {self.max_retries} attempts: {last_exception}")

    def generate(self, prompt: str) -> str:
        """Generate a completion for a single assembled prompt string."""
        return self.chat([{"role": "user", "content": prompt}])

    def chat(self, messages: list[ChatMessage]) -> str:
        """Generate a completion for a full message history."""
        return self._call_api(messages)

    def stream(self, messages: list[ChatMessage]) -> Iterator[str]:
        """Stream a completion using server-sent events.

        Falls back to a single non-streaming call (yielded as one
        chunk) if the streaming request fails, so callers can rely on
        ``stream()`` always producing output rather than crashing.
        """
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }

        try:
            with self._session.post(
                url, json=payload, timeout=self.timeout, stream=True
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                        delta = event["choices"][0].get("delta", {}).get("content")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except (requests.exceptions.RequestException, LLMError) as exc:
            logger.warning("Streaming failed (%s); falling back to non-streaming chat", exc)
            yield self.chat(messages)
