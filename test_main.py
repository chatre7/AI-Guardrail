
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

# Import the FastAPI app instance from your main application file
from main import app, FORBIDDEN_KEYWORDS, GUARDRAIL_SYSTEM_PROMPT, JUDGE_MODEL, CHAT_MODEL

# Pytest marker for all tests in this file to be treated as asyncio
pytestmark = pytest.mark.asyncio


# Helper to create a mock Ollama response chunk
def create_chat_chunk(content: str):
    return {"message": {"content": content}}

# Helper to create a mock Ollama generation response
def create_generate_response(content: str):
    return {"response": content}

@pytest.fixture
def mock_ollama_client():
    """Provides a fixture to patch the ollama.AsyncClient for all tests."""
    with patch("main.ollama.AsyncClient", new_callable=AsyncMock) as mock_client:
        yield mock_client


async def test_l1_keyword_block_in_prompt():
    """
    Test Case: (Layer 1) The user's prompt contains a forbidden keyword.
    Expected: The stream should be immediately rejected with the specific L1 message.
    """
    keyword = FORBIDDEN_KEYWORDS[0]
    prompt = f"Tell me about {keyword}"

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(f"/chat?prompt={prompt}")

    assert response.status_code == 200
    assert "สามารถให้ข้อมูลและช่วยเหลือเฉพาะเรื่องบริการของเราเท่านั้น" in response.text


async def test_l2_judge_block_in_prompt(mock_ollama_client: MagicMock):
    """
    Test Case: (Layer 2) The prompt is clean, but the AI Judge deems it unsafe.
    Expected: The stream should be rejected with the L2 message before the chat model is called.
    """
    # Configure the judge model to return "UNSAFE"
    mock_judge_response = create_generate_response("UNSAFE")
    mock_ollama_client.return_value.generate.return_value = mock_judge_response

    prompt = "A prompt that the judge will consider unsafe"
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(f"/chat?prompt={prompt}")

    # Assertions
    assert response.status_code == 200
    assert "เนื้อหานี้ไม่สอดคล้องกับนโยบายการให้บริการ" in response.text
    
    # Verify that the judge was called but the chat model was not
    mock_ollama_client.return_value.generate.assert_called_once()
    mock_ollama_client.return_value.chat.assert_not_called()


async def test_successful_stream_happy_path(mock_ollama_client: MagicMock):
    """
    Test Case: (Happy Path) The prompt and the entire response are safe.
    Expected: The full response from the chat model should be streamed successfully.
    """
    # 1. Configure Judge to always be SAFE
    mock_judge_response = create_generate_response("SAFE")
    mock_ollama_client.return_value.generate.return_value = mock_judge_response

    # 2. Configure Chat model to stream a predefined response
    async def mock_chat_stream():
        yield create_chat_chunk("Hello, ")
        yield create_chat_chunk("welcome to ")
        yield create_chat_chunk("SkyHigh Airlines.")
    
    # Use a separate mock for the chat stream since it's an async context manager
    chat_stream_mock = AsyncMock()
    chat_stream_mock.__aenter__.return_value = mock_chat_stream()
    mock_ollama_client.return_value.chat.return_value = chat_stream_mock

    # 3. Make the request
    prompt = "A perfectly safe prompt"
    full_response = ""
    async with AsyncClient(app=app, base_url="http://test") as client:
        async with client.stream("GET", f"/chat?prompt={prompt}") as response:
            async for chunk in response.aiter_text():
                full_response += chunk

    # 4. Assertions
    assert response.status_code == 200
    assert "Hello, welcome to SkyHigh Airlines." in full_response
    assert "UNSAFE" not in full_response
    assert "ขออภัย" not in full_response
    
    # Verify judge was called (at least once for the prompt) and chat was called
    mock_ollama_client.return_value.generate.assert_called()
    mock_ollama_client.return_value.chat.assert_called_once()


async def test_l3_response_violation_and_termination(mock_ollama_client: MagicMock):
    """
    Test Case: (Layer 3) The prompt is safe, but the LLM response contains an unsafe word.
    Expected: The stream begins, then terminates with the L3 message when the violation is detected.
    """
    # 1. Configure Judge to be SAFE for the prompt, but UNSAFE for a specific word
    def judge_side_effect(*args, **kwargs):
        prompt_text = kwargs.get("prompt", "")
        if "A safe prompt" in prompt_text:
            return asyncio.sleep(0.01, create_generate_response("SAFE"))
        elif "secret" in prompt_text:
            return asyncio.sleep(0.01, create_generate_response("UNSAFE"))
        else:
            return asyncio.sleep(0.01, create_generate_response("SAFE"))
    
    mock_ollama_client.return_value.generate.side_effect = judge_side_effect

    # 2. Configure Chat model to stream a response containing the unsafe word
    async def mock_chat_stream():
        yield create_chat_chunk("Here is the information. ")
        yield create_chat_chunk("The secret code is 1234.") # This will trigger the judge

    chat_stream_mock = AsyncMock()
    chat_stream_mock.__aenter__.return_value = mock_chat_stream()
    mock_ollama_client.return_value.chat.return_value = chat_stream_mock

    # 3. Make the request and collect the response
    prompt = "A safe prompt"
    full_response = ""
    async with AsyncClient(app=app, base_url="http://test") as client:
        async with client.stream("GET", f"/chat?prompt={prompt}") as response:
            async for chunk in response.aiter_text():
                full_response += chunk
    
    # 4. Assertions
    assert response.status_code == 200
    # Check that the initial safe part was streamed
    assert "Here is the information." in full_response
    # Check that the stream was terminated with the correct L3 message
    assert "เนื้อหาที่กำลังสนทนาไม่เป็นไปตามนโยบาย" in full_response
    # Check that the unsafe content itself might or might not be there depending on timing,
    # but the termination message is key.


async def test_judge_timeout_fails_open(mock_ollama_client: MagicMock):
    """
    Test Case: The judge model times out during validation.
    Expected: The system should "fail-open", assuming the content is safe and continuing the stream.
    """
    # 1. Configure Judge to simulate a timeout
    mock_ollama_client.return_value.generate.side_effect = asyncio.TimeoutError

    # 2. Configure Chat model for a normal stream
    async def mock_chat_stream():
        yield create_chat_chunk("This should stream successfully.")
    
    chat_stream_mock = AsyncMock()
    chat_stream_mock.__aenter__.return_value = mock_chat_stream()
    mock_ollama_client.return_value.chat.return_value = chat_stream_mock
    
    # 3. Make the request
    prompt = "A safe prompt"
    full_response = ""
    async with AsyncClient(app=app, base_url="http://test") as client:
        async with client.stream("GET", f"/chat?prompt={prompt}") as response:
            async for chunk in response.aiter_text():
                full_response += chunk

    # 4. Assertions
    assert "This should stream successfully." in full_response
    assert "ขออภัย" not in full_response


async def test_ollama_connection_error(mock_ollama_client: MagicMock):
    """
    Test Case: The application cannot connect to the Ollama service.
    Expected: A user-friendly error message is streamed.
    """
    # 1. Configure the client to raise an error
    mock_ollama_client.return_value.chat.side_effect = Exception("Connection failed")

    # 2. Make the request
    prompt = "Any prompt"
    full_response = ""
    async with AsyncClient(app=app, base_url="http://test") as client:
        async with client.stream("GET", f"/chat?prompt={prompt}") as response:
            async for chunk in response.aiter_text():
                full_response += chunk

    # 3. Assertions
    assert "ระบบกำลังมีปัญหาชั่วคราว" in full_response

