"""
Integration test suite for the SkyHigh Airlines Chatbot.

These tests run against a live test client of the FastAPI application,
ensuring that middleware (CORS, Rate Limiting) and dependency-injected
security features (API Keys) are working as expected.
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from main import API_KEY, API_KEY_NAME, MAX_PROMPT_LENGTH, app

# Pytest marker for all tests in this file to be treated as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def api_headers() -> Dict[str, str]:
    """Provides valid API headers for authenticated requests."""
    return {API_KEY_NAME: API_KEY}


@pytest.fixture(scope="module")
def invalid_api_headers() -> Dict[str, str]:
    """Provides invalid API headers for testing authentication failure."""
    return {API_KEY_NAME: "invalid-key"}


@pytest.mark.asyncio
async def test_missing_api_key_fails() -> None:
    """
    Tests that a request without an API key is rejected.

    Expected:
        HTTP 401 Unauthorized error.
    """
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/chat?prompt=test")
    assert response.status_code == 401
    assert "Invalid or missing API Key" in response.json()["detail"]


@pytest.mark.asyncio
async def test_invalid_api_key_fails(invalid_api_headers: Dict[str, str]) -> None:
    """
    Tests that a request with an invalid API key is rejected.

    Expected:
        HTTP 401 Unauthorized error.
    """
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/chat?prompt=test", headers=invalid_api_headers)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_prompt_too_long_fails(api_headers: Dict[str, str]) -> None:
    """
    Tests that a prompt exceeding the maximum length is rejected.

    Expected:
        HTTP 413 Payload Too Large error.
    """
    long_prompt = "a" * (MAX_PROMPT_LENGTH + 1)
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(f"/chat?prompt={long_prompt}", headers=api_headers)
    assert response.status_code == 413
    assert "Prompt exceeds maximum length" in response.json()["detail"]


@pytest.mark.asyncio
async def test_rate_limiting_works(api_headers: Dict[str, str]) -> None:
    """
    Tests that the rate limiter blocks excessive requests.
    Note: The limiter is configured in memory for tests.

    Expected:
        After exceeding the limit (10/minute), subsequent requests should
        receive an HTTP 429 Too Many Requests error.
    """
    # Mock Ollama to prevent actual calls during the rate limit test
    with patch("main.ollama.AsyncClient", new_callable=AsyncMock) as mock_client:
        mock_client.return_value.generate.return_value = {"response": "SAFE"}

        async def mock_chat_stream(*args: Any, **kwargs: Any) -> AsyncMock:
            stream = AsyncMock()
            stream.__aenter__.return_value = iter(
                [{"message": {"content": "ok"}}]
            )
            return stream
        mock_client.return_value.chat.return_value = mock_chat_stream()

        async with AsyncClient(app=app, base_url="http://test") as client:
            # Send 10 successful requests to hit the limit
            for i in range(10):
                response = await client.get("/chat?prompt=test", headers=api_headers)
                assert response.status_code == 200, f"Request {i+1} failed"

            # The 11th request should be rate-limited
            response = await client.get("/chat?prompt=test", headers=api_headers)
            assert response.status_code == 429


@pytest.mark.asyncio
async def test_cors_headers_present(api_headers: Dict[str, str]) -> None:
    """
    Tests that CORS headers are present in the response.

    Expected:
        'access-control-allow-origin' header should be present, matching
        one of the configured origins.
    """
    # Mock Ollama to ensure a quick, successful response
    with patch("main.ollama.AsyncClient", new_callable=AsyncMock):
        origin = "http://localhost:3000"
        headers = {**api_headers, "Origin": origin}
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/chat?prompt=test", headers=headers)

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
