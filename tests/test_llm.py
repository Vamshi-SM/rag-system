"""Unit tests for src/llm (Phase 8)."""

from __future__ import annotations

import pytest

from src.llm.base_llm import LLMError
from src.llm.sllm import SLLMChat


class TestSLLMChat:
    def test_generate_returns_string(self, llm: SLLMChat) -> None:
        answer = llm.generate("What is the refund policy?")
        assert isinstance(answer, str)
        assert answer

    def test_chat_passes_full_message_history(self, llm: SLLMChat) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        answer = llm.chat(messages)
        assert isinstance(answer, str)
        assert answer

    def test_stream_falls_back_to_full_response_by_default(self, llm: SLLMChat) -> None:
        chunks = list(llm.stream([{"role": "user", "content": "Hi"}]))
        assert len(chunks) == 1
        assert isinstance(chunks[0], str)

    def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chat = SLLMChat(
            api_key="k", base_url="https://fake/v1", model="m", max_retries=3, backoff_factor=0.01
        )
        attempts = {"count": 0}

        class FakeResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                # Ollama-format response (matches SharedLLM /api/chat endpoint)
                return {"model": "m", "message": {"role": "assistant", "content": "OK"}, "done": True}

        def flaky_post(url, json, timeout):  # noqa: ANN001
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise __import__("requests").exceptions.ConnectionError("simulated network blip")
            return FakeResponse()

        monkeypatch.setattr(chat._session, "post", flaky_post)

        answer = chat.generate("hello")
        assert answer == "OK"
        assert attempts["count"] == 2

    def test_raises_llm_error_after_exhausting_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chat = SLLMChat(
            api_key="k", base_url="https://fake/v1", model="m", max_retries=2, backoff_factor=0.01
        )

        def always_fails(url, json, timeout):  # noqa: ANN001
            raise __import__("requests").exceptions.ConnectionError("simulated outage")

        monkeypatch.setattr(chat._session, "post", always_fails)

        with pytest.raises(LLMError):
            chat.generate("hello")

    def test_retries_on_5xx_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chat = SLLMChat(
            api_key="k", base_url="https://fake/v1", model="m", max_retries=2, backoff_factor=0.01
        )

        class ServerErrorResponse:
            status_code = 503
            text = "service unavailable"

        def flaky_post(url, json, timeout):  # noqa: ANN001
            return ServerErrorResponse()

        monkeypatch.setattr(chat._session, "post", flaky_post)

        with pytest.raises(LLMError):
            chat.generate("hello")
