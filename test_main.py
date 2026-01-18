"""
Test suite for the SkyHigh Airlines Chatbot main application.

This suite uses pytest, httpx, and unittest.mock to test the FastAPI app's
endpoints and guardrail logic. It focuses on unit tests where external
services like Ollama are mocked.
"""

import asyncio
from typing import Any, AsyncGenerator, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from main import (
    API_KEY,
    API_KEY_NAME,
    CHAT_MODEL,
    FORBIDDEN_KEYWORDS,
    JUDGE_MODEL,
    MAX_PROMPT_LENGTH,
    app,
)

# Pytest marker for all tests in this file to be treated as asyncio
pytestmark = pytest.mark.asyncio


# ==========================================
# Helper Functions & Fixtures
# ==========================================


def create_chat_chunk(content: str) -> Dict[str, Any]:
    """Creates a mock Ollama chat response chunk."""
    return {"message": {"content": content}}


def create_generate_response(content: str) -> Dict[str, Any]:
    """Creates a mock Ollama generate response."""
    return {"response": content}


@pytest.fixture
def mock_ollama_client() -> MagicMock:
    """
    Provides a fixture that patches `main.ollama.AsyncClient` for all tests.
    This prevents real calls to the Ollama service during unit testing.
    """
    with patch("main.ollama.AsyncClient", new_callable=AsyncMock) as mock_client:
        yield mock_client


@pytest.fixture
def api_headers() -> Dict[str, str]:
    """Provides valid API headers for authenticated requests."""
    return {API_KEY_NAME: API_KEY}


# ==========================================
# Test Cases
# ==========================================


async def test_l1_keyword_block_in_prompt(api_headers: Dict[str, str]) -> None:
    """
    Tests L1 Guardrail: A prompt with a forbidden keyword should be blocked.

    Scenario:
        The user's prompt contains a word from the `FORBIDDEN_KEYWORDS` list.
    Expected:
        The API should return an immediate rejection message specific to L1
        and not call any LLM.
    """
    keyword = FORBIDDEN_KEYWORDS[0]
    prompt = f"Tell me about {keyword}"

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(f"/chat?prompt={prompt}", headers=api_headers)

    assert response.status_code == 200
    assert "ให้ข้อมูลเฉพาะบริการของเราเท่านั้น" in response.text


async def test_l2_judge_block_in_prompt(
    mock_ollama_client: MagicMock, api_headers: Dict[str, str]
) -> None:
    """
    Tests L2 Guardrail: A prompt deemed unsafe by the AI Judge should be blocked.

    Scenario:
        The user's prompt is free of keywords but is classified as "UNSAFE"
        by the JUDGE_MODEL.
    Expected:
        The API should return a rejection message specific to L2. The judge model
        should be called, but the chat model should not.
    """
    mock_judge_response = create_generate_response("UNSAFE")
    mock_ollama_client.return_value.generate.return_value = mock_judge_response

    prompt = "A prompt that the judge will consider unsafe"
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(f"/chat?prompt={prompt}", headers=api_headers)

    assert response.status_code == 200
    assert "เนื้อหานี้ไม่สอดคล้องกับนโยบายของเรา" in response.text
    mock_ollama_client.return_value.generate.assert_called_once()
    mock_ollama_client.return_value.chat.assert_not_called()


async def test_successful_stream_happy_path(
    mock_ollama_client: MagicMock, api_headers: Dict[str, str]
) -> None:
    """
    Tests the Happy Path: A safe prompt and response should stream successfully.

    Scenario:
        The prompt is safe, and the AI Judge finds no issues with the
        generated response.
    Expected:
        The full, uninterrupted response from the chat model is streamed to the user.
    """
    mock_ollama_client.return_value.generate.return_value = create_generate_response(
        "SAFE"
    )

    async def mock_chat_stream() -> AsyncGenerator[Dict[str, Any], None]:
        yield create_chat_chunk("Hello, ")
        yield create_chat_chunk("welcome.")

    chat_stream_mock = AsyncMock()
    chat_stream_mock.__aenter__.return_value = mock_chat_stream()
    mock_ollama_client.return_value.chat.return_value = chat_stream_mock

    prompt = "A perfectly safe prompt"
    full_response = ""
    async with AsyncClient(app=app, base_url="http://test") as client:
        async with client.stream(
            "GET", f"/chat?prompt={prompt}", headers=api_headers
        ) as response:
            async for chunk in response.aiter_text():
                full_response += chunk

    assert response.status_code == 200
    assert "Hello, welcome." in full_response
    assert "ขออภัย" not in full_response
    mock_ollama_client.return_value.generate.assert_called()
    mock_ollama_client.return_value.chat.assert_called_once()


async def test_l3_response_violation_and_termination(
    mock_ollama_client: MagicMock, api_headers: Dict[str, str]
) -> None:
    """
    Tests L3 Guardrail: A stream should be terminated if the response is unsafe.

    Scenario:
        The prompt is safe, but during streaming, the chat model generates a
        response chunk that the AI Judge deems "UNSAFE".
    Expected:
        The stream starts normally, but as soon as the unsafe content is
        detected, the stream is terminated with an L3 rejection message.
    """

    async def judge_side_effect(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        prompt_text = kwargs.get("prompt", "")
        await asyncio.sleep(0)  # Yield control to the event loop
        if "secret" in prompt_text:
            return create_generate_response("UNSAFE")
        return create_generate_response("SAFE")

    mock_ollama_client.return_value.generate.side_effect = judge_side_effect

    async def mock_chat_stream() -> AsyncGenerator[Dict[str, Any], None]:
        yield create_chat_chunk("Here is the info. ")
        yield create_chat_chunk("The secret is 1234.")

    chat_stream_mock = AsyncMock()
    chat_stream_mock.__aenter__.return_value = mock_chat_stream()
    mock_ollama_client.return_value.chat.return_value = chat_stream_mock

    prompt = "A safe prompt"
    full_response = ""
    async with AsyncClient(app=app, base_url="http://test") as client:
        async with client.stream(
            "GET", f"/chat?prompt={prompt}", headers=api_headers
        ) as response:
            async for chunk in response.aiter_text():
                full_response += chunk

    assert response.status_code == 200
    assert "Here is the info." in full_response
    assert "การสนทนาไม่เป็นไปตามนโยบาย" in full_response


async def test_judge_timeout_fails_open(
    mock_ollama_client: MagicMock, api_headers: Dict[str, str]
) -> None:
    """
    Tests fail-open behavior when the AI Judge times out.

    Scenario:
        The call to the JUDGE_MODEL raises an `asyncio.TimeoutError`.
    Expected:
        The system should not block the stream. It should "fail-open" and
        continue streaming the response as if it were safe.
    """
    mock_ollama_client.return_value.generate.side_effect = asyncio.TimeoutError

    async def mock_chat_stream() -> AsyncGenerator[Dict[str, Any], None]:
        yield create_chat_chunk("This should stream successfully.")

    chat_stream_mock = AsyncMock()
    chat_stream_mock.__aenter__.return_value = mock_chat_stream()
    mock_ollama_client.return_value.chat.return_value = chat_stream_mock

    prompt = "A safe prompt"
    full_response = ""
    async with AsyncClient(app=app, base_url="http://test") as client:
        async with client.stream(
            "GET", f"/chat?prompt={prompt}", headers=api_headers
        ) as response:
            async for chunk in response.aiter_text():
                full_response += chunk

    assert "This should stream successfully." in full_response
    assert "ขออภัย" not in full_response


async def test_ollama_connection_error_streams_error_message(
    mock_ollama_client: MagicMock, api_headers: Dict[str, str]
) -> None:
    """
    Tests the error handling when the Ollama service is unreachable.

    Scenario:
        The call to `ollama.AsyncClient.chat` raises a generic exception.
    Expected:
        The API should stream a user-friendly error message.
    """
    mock_ollama_client.return_value.chat.side_effect = Exception("Connection failed")

    prompt = "Any prompt"
    full_response = ""
    async with AsyncClient(app=app, base_url="http://test") as client:
        async with client.stream(
            "GET", f"/chat?prompt={prompt}", headers=api_headers
        ) as response:
            async for chunk in response.aiter_text():
                full_response += chunk

    assert "ระบบมีปัญหาชั่วคราว" in full_response