"""Chat/completion provider backed by the SharedLLM remote API.

The SharedLLM chat endpoint is Ollama-compatible::

    POST https://api.sharedllm.com/ollama/api/chat

    Request:
        {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false
        }

    Response:
        {
            "model": "deepseek-v4-flash",
            "message": {"role": "assistant", "content": "..."},
            "done": true
        }

Authentication uses the ``X-SharedLLM-Key`` header.

If the API shape changes, only ``_call_api`` and ``_stream_api`` need
updating — the retry, timeout, and back-off logic is self-contained and
reusable as-is.  This mirrors the structure of the embedding providers so
both provider layers are easy to read side-by-side.
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
    """Chat/completion provider backed by the SharedLLM API.

    Implements the :class:`~src.llm.base_llm.BaseLLM` interface so it can
    be dropped into the RAG pipeline without any changes upstream.

    Args:
        api_key:        SharedLLM API key — set via ``SHAREDLLM_API_KEY``
                        (or legacy ``SLLM_API_KEY``).  Sent as the
                        ``X-SharedLLM-Key`` request header.
        base_url:       Base URL of the SharedLLM Ollama-compat endpoint,
                        e.g. ``https://api.sharedllm.com/ollama``.
                        Set via ``SHAREDLLM_CHAT_BASE_URL``.
        model:          Chat model name, e.g. ``deepseek-v4-flash``.
                        Set via ``CHAT_MODEL``.
        timeout:        Per-request timeout in seconds.  Defaults to 60 s.
        max_retries:    Number of retry attempts on transient failures.
        backoff_factor: Base delay (seconds) for exponential back-off
                        between retries.  Actual delay on attempt *n* is
                        ``backoff_factor * 2 ** (n - 1)``.
        temperature:    Sampling temperature forwarded to the model.
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
            logger.warning(
                "SHAREDLLM_API_KEY is empty — chat requests will be rejected by the API"
            )

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_api(self, messages: list[ChatMessage]) -> str:
        """POST to the chat endpoint with exponential-backoff retries.

        Retries on:
        - Network-level failures (``ConnectionError``, ``Timeout``)
        - HTTP 429 (rate-limit) and 5xx (server error) responses

        Raises:
            LLMError: after exhausting all retry attempts.
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

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.post(url, json=payload, timeout=self.timeout)
                print("\n========== SHAREDLLM ==========")
                print("URL      :", url)
                print("MODEL    :", self.model)
                print("API KEY  :", self.api_key[:20] + "...")
                print("===============================\n")

                response = self._session.post(url, json=payload, timeout=self.timeout)

                print("HTTP STATUS:", response.status_code)
                # Treat rate-limit and server errors as retryable.
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.exceptions.HTTPError(
                        f"Retryable status {response.status_code}: {response.text[:200]}"
                    )

                response.raise_for_status()
                body = response.json()

                # Ollama-format response: {"message": {"role": "...", "content": "..."}, ...}
                if "message" not in body:
                    raise LLMError(
                        f"SharedLLM response missing 'message' key. "
                        f"Got keys: {list(body.keys())}"
                    )

                print("Response model:", body.get("model"))
                print("DONE:", body.get("done"))
                return body["message"]["content"]

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                KeyError,
                IndexError,
                ValueError,
                LLMError,
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
                        "Chat API call failed after %d attempts: %s",
                        self.max_retries,
                        exc,
                    )

        raise LLMError(
            f"Failed to get chat completion after {self.max_retries} attempts: {last_exception}"
        )

    def _stream_api(self, messages: list[ChatMessage]) -> Iterator[str]:
        """POST with ``stream: true`` and yield content tokens as they arrive.

        Parses Ollama-format streaming responses::

            {"message": {"role": "assistant", "content": "<token>"}, "done": false}
            {"message": {"role": "assistant", "content": ""}, "done": true}

        Raises:
            requests.exceptions.RequestException: propagated to the caller
                of :meth:`stream` which converts it to a fallback.
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
        print("\n========== SHAREDLLM ==========")
        print("URL   :", url)
        print("MODEL :", self.model)
        print("===============================\n")

        with self._session.post(
            url, json=payload, timeout=self.timeout, stream=True
        ) as response:
            print("HTTP STATUS:", response.status_code)
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if event.get("done"):
                        break
                    message = event.get("message") or {}
                    content = message.get("content")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Malformed line — skip and continue streaming.
                    continue

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Generate a completion for a single assembled prompt string.

        Wraps the prompt as a ``user`` message and delegates to
        :meth:`chat`.
        """
        return self.chat([{"role": "user", "content": prompt}])

    def chat(self, messages: list[ChatMessage]) -> str:
        """Generate a completion for a full multi-turn message history."""
        return self._call_api(messages)

    def stream(self, messages: list[ChatMessage]) -> Iterator[str]:
        """Stream a completion token-by-token using Ollama chunked responses.

        Falls back to a single non-streaming :meth:`chat` call (yielded as
        one chunk) if the streaming request fails, so callers can rely on
        ``stream()`` always producing at least one output chunk.
        """
        try:
            yield from self._stream_api(messages)
        except (requests.exceptions.RequestException, LLMError) as exc:
            logger.warning(
                "Streaming failed (%s); falling back to non-streaming chat", exc
            )
            yield self.chat(messages)


# ---------------------------------------------------------------------------
# Alias — kept for backwards compatibility with scripts that import by this name
# ---------------------------------------------------------------------------
SharedLLMChat = SLLMChat
