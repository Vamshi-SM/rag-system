"""Tests for src/llm/sharedllm_chat.py (SLLMChat / SharedLLMChat).

Offline unit tests (default)
-----------------------------
These run without any network access by monkey-patching ``_call_api`` at
the seam used by the rest of the test suite (see conftest.py).

Live integration test (opt-in)
-------------------------------
The ``test_live_chat`` test requires:
  - A valid ``SHAREDLLM_API_KEY`` (or ``SLLM_API_KEY``) in .env
  - Network access to https://api.sharedllm.com

Run it explicitly with::

    pytest tests/test_sharedllm.py -v -s -m integration

or skip all network tests with::

    pytest tests/test_sharedllm.py -v -s -m "not integration"
"""

from __future__ import annotations

import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import pytest
import requests

# Allow running this file directly: `python tests/test_sharedllm.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.base_llm import LLMError
from src.llm.sllm import SharedLLMChat, SLLMChat  # canonical import path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chat(
    api_key: str = "test-key",
    base_url: str = "https://fake-sharedllm.test/ollama",
    model: str = "deepseek-v4-flash",
    **kwargs,
) -> SLLMChat:
    return SLLMChat(api_key=api_key, base_url=base_url, model=model, **kwargs)


class _OkResponse:
    """Minimal fake of a successful requests.Response."""

    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"model": "deepseek-v4-flash", "message": {"role": "assistant", "content": "OK"}, "done": True}


class _ServerErrorResponse:
    status_code = 503
    text = "service unavailable"


# ---------------------------------------------------------------------------
# Offline unit tests
# ---------------------------------------------------------------------------


class TestSLLMChatImport:
    """Verify the import aliases work correctly."""

    def test_sllm_chat_importable(self) -> None:
        assert SLLMChat is not None

    def test_shared_llm_chat_alias_is_same_class(self) -> None:
        """SharedLLMChat must be the exact same class as SLLMChat."""
        assert SharedLLMChat is SLLMChat


class TestSLLMChatInit:
    def test_empty_api_key_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        with caplog.at_level(logging.WARNING, logger="src.llm.sharedllm_chat"):
            _make_chat(api_key="")
        assert any("API_KEY" in r.message or "empty" in r.message for r in caplog.records)

    def test_session_has_auth_header(self) -> None:
        chat = _make_chat(api_key="secret-key")
        assert chat._session.headers.get("X-SharedLLM-Key") == "secret-key"

    def test_session_has_content_type_header(self) -> None:
        chat = _make_chat()
        assert "application/json" in chat._session.headers.get("Content-Type", "")

    def test_base_url_trailing_slash_stripped(self) -> None:
        chat = _make_chat(base_url="https://api.sharedllm.com/ollama/")
        assert chat.base_url == "https://api.sharedllm.com/ollama"


class TestSLLMChatGenerate:
    def test_generate_wraps_prompt_as_user_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chat = _make_chat()
        captured: list[list[dict]] = []

        def fake_call_api(self, messages):  # noqa: ANN001
            captured.append(messages)
            return "hello"

        monkeypatch.setattr(SLLMChat, "_call_api", fake_call_api)
        result = chat.generate("what is two plus two?")
        assert result == "hello"
        assert captured[0] == [{"role": "user", "content": "what is two plus two?"}]

    def test_generate_returns_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chat = _make_chat()
        monkeypatch.setattr(SLLMChat, "_call_api", lambda self, msgs: "42")
        assert isinstance(chat.generate("any prompt"), str)


class TestSLLMChatCallApi:
    def test_succeeds_on_first_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chat = _make_chat()
        monkeypatch.setattr(chat._session, "post", lambda *a, **kw: _OkResponse())
        assert chat._call_api([{"role": "user", "content": "hi"}]) == "OK"

    def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chat = _make_chat(max_retries=3, backoff_factor=0.0)
        calls = {"n": 0}

        def flaky_post(url, json, timeout):  # noqa: ANN001
            calls["n"] += 1
            if calls["n"] < 2:
                raise requests.exceptions.ConnectionError("network blip")
            return _OkResponse()

        monkeypatch.setattr(chat._session, "post", flaky_post)
        assert chat._call_api([{"role": "user", "content": "hi"}]) == "OK"
        assert calls["n"] == 2

    def test_raises_llm_error_after_exhausting_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chat = _make_chat(max_retries=2, backoff_factor=0.0)
        monkeypatch.setattr(
            chat._session,
            "post",
            lambda *a, **kw: (_ for _ in ()).throw(
                requests.exceptions.ConnectionError("always fails")
            ),
        )
        with pytest.raises(LLMError, match="Failed to get chat completion"):
            chat._call_api([{"role": "user", "content": "hi"}])

    def test_retries_on_5xx_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chat = _make_chat(max_retries=2, backoff_factor=0.0)
        monkeypatch.setattr(chat._session, "post", lambda *a, **kw: _ServerErrorResponse())
        with pytest.raises(LLMError):
            chat._call_api([{"role": "user", "content": "hi"}])

    def test_raises_llm_error_on_missing_message_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class BadResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"done": True}  # missing "message" key

        chat = _make_chat(max_retries=1, backoff_factor=0.0)
        monkeypatch.setattr(chat._session, "post", lambda *a, **kw: BadResponse())
        with pytest.raises(LLMError, match="missing 'message' key"):
            chat._call_api([{"role": "user", "content": "hi"}])


class TestSLLMChatStream:
    def test_stream_falls_back_when_request_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chat = _make_chat()

        def fail_stream(self, messages):  # noqa: ANN001
            raise requests.exceptions.ConnectionError("streaming unavailable")

        monkeypatch.setattr(SLLMChat, "_stream_api", fail_stream)
        monkeypatch.setattr(SLLMChat, "_call_api", lambda self, msgs: "fallback answer")

        chunks = list(chat.stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["fallback answer"]

    def test_stream_yields_content_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chat = _make_chat()

        def fake_stream_api(self, messages):  # noqa: ANN001
            yield "Hello"
            yield ", world"
            yield "!"

        monkeypatch.setattr(SLLMChat, "_stream_api", fake_stream_api)
        chunks = list(chat.stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["Hello", ", world", "!"]


# ---------------------------------------------------------------------------
# Live integration test (opt-in, requires real API key + network)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_live_chat() -> None:
    """End-to-end smoke test against the real SharedLLM API.

    Requires SHAREDLLM_API_KEY (or SLLM_API_KEY) to be set in .env.
    """
    from src.config import settings

    api_key = settings.sllm_api_key
    if not api_key:
        pytest.skip("SHAREDLLM_API_KEY not set — skipping live test")

    chat = SLLMChat(
        api_key=api_key,
        base_url=settings.sllm_base_url,
        model=settings.chat_model,
        timeout=30.0,
        max_retries=1,
    )
    answer = chat.generate("Reply with exactly three words: I am ready.")
    assert isinstance(answer, str), f"Expected str, got {type(answer)}"
    assert len(answer.strip()) > 0, "Expected non-empty answer"
    print(f"\n[live] SharedLLM response: {answer!r}")


# ---------------------------------------------------------------------------
# Allow direct execution: python tests/test_sharedllm.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.config import settings

    print("Running live SharedLLM smoke test...")
    chat = SLLMChat(
        api_key=settings.sllm_api_key,
        base_url=settings.sllm_base_url,
        model=settings.chat_model,
        timeout=30.0,
        max_retries=2,
    )
    response = chat.generate("Hello! Reply in one sentence.")
    print(f"Model response: {response}")