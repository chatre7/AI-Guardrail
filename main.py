"""
Main application file for the SkyHigh Airlines Chatbot.

This module sets up a FastAPI application with a multi-layered, asynchronous
guardrail system for an Ollama-powered chatbot. It includes rate limiting,
API key authentication, CORS, and input validation.
"""

import asyncio
import logging
import os
from typing import AsyncGenerator, Dict, List, Set

import ollama
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ==========================================
# Logging & Configuration
# ==========================================
logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger("skyhigh-airlines-chatbot")

# --- Security & Limits Configuration ---
API_KEY: str = os.getenv("API_KEY", "skyhigh-secret-key-for-dev")
API_KEY_NAME: str = "X-API-KEY"
MAX_PROMPT_LENGTH: int = 1000

# --- Rate Limiting Configuration ---
limiter: Limiter = Limiter(key_func=get_remote_address)

# --- CORS Configuration ---
origins: List[str] = [
    "http://localhost",
    "http://localhost:3000",  # Example for a React frontend
    "http://localhost:8080",  # Example for a Vue frontend
]

# --- Application Setup ---
app: FastAPI = FastAPI(title="SkyHigh Airlines Chatbot (Hardened API)")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# API Key Authentication
# ==========================================
api_key_header: APIKeyHeader = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


async def get_api_key(api_key_header_value: str = Security(api_key_header)) -> str:
    """
    Validates the API key provided in the request header.

    Args:
        api_key_header_value: The API key extracted from the header.

    Raises:
        HTTPException: If the API key is invalid or missing.

    Returns:
        The validated API key.
    """
    if api_key_header_value == API_KEY:
        return api_key_header_value
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )


# ==========================================
# Models, Prompts, and Guardrail Logic
# ==========================================
CHAT_MODEL: str = "gpt-oss:20b-cloud"
JUDGE_MODEL: str = "qwen3:4b"
CHECK_INTERVAL: int = 40
JUDGE_TIMEOUT_SEC: float = 5.0
MAX_CONCURRENT_JUDGES: int = 1
judge_semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT_JUDGES)

FORBIDDEN_KEYWORDS: List[str] = [
    "airasia",
    "แอร์เอเชีย",
    "nok air",
    "นกแอร์",
    "lion air",
    "ไลอ้อนแอร์",
    "vietjet",
    "qatar airways",
    "emirates",
    "singapore airlines",
    "การบินไทย",
]

GUARDRAIL_SYSTEM_PROMPT: str = """
คุณคือนักตรวจสอบเนื้อหาที่เข้มงวดมากสำหรับแชทบอทบริการลูกค้าของสายการบิน SkyHigh Airlines
ตอบได้แค่คำเดียว: SAFE หรือ UNSAFE

ถือว่า UNSAFE ทันที หากข้อความมี ANY อย่างใดอย่างหนึ่ง ดังนี้:
- การเมือง, รัฐบาล, ประท้วง, การเลือกตั้ง
- ชื่อสายการบินใด ๆ ที่ไม่ใช่ SkyHigh Airlines
- ศาสนา, พระ, พระเจ้า (ยกเว้นคำทั่วไปเช่น สงกรานต์)

ข้อความที่ต้องตรวจ: "{text}"
ตอบแค่ SAFE หรือ UNSAFE เท่านั้น
"""


async def run_guardrail_judge(
    text: str,
    is_safe_event: asyncio.Event,
    request_id: str = "",
) -> None:
    """
    Runs a check on the given text using the JUDGE_MODEL to determine if it's safe.

    If the content is judged as "UNSAFE", it clears the `is_safe_event`,
    triggering the application's circuit breaker. This function uses a semaphore
    to limit concurrent executions.

    Args:
        text: The text chunk to validate.
        is_safe_event: The event to clear if the content is unsafe.
        request_id: A unique identifier for logging purposes.
    """
    if not text.strip() or judge_semaphore.locked():
        if judge_semaphore.locked():
            logger.debug(f"[{request_id}] Judge busy, skipping check.")
        return

    async with judge_semaphore:
        try:
            async with asyncio.timeout(JUDGE_TIMEOUT_SEC):
                prompt = GUARDRAIL_SYSTEM_PROMPT.format(text=text)
                client = ollama.AsyncClient()
                resp = await client.generate(
                    model=JUDGE_MODEL, prompt=prompt, stream=False
                )
                if "UNSAFE" in resp["response"].strip().upper():
                    logger.warning(f"[{request_id}] UNSAFE detected: ...{text[-60:]}")
                    is_safe_event.clear()
        except asyncio.TimeoutError:
            logger.warning(f"[{request_id}] Judge timeout → fail-open (assuming SAFE).")
        except Exception as e:
            logger.error(f"[{request_id}] Judge error: {e}", exc_info=True)


# ==========================================
# Hardened API Endpoint
# ==========================================
@app.get("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(
    request: Request,
    prompt: str,
    api_key: str = Depends(get_api_key),
) -> StreamingResponse:
    """
    Main chat endpoint with multi-layered security and guardrails.

    - Authenticates the request via API key.
    - Validates and sanitizes the input prompt.
    - Applies Layer 1 (keyword) and Layer 2 (AI prompt check) guardrails.
    - Streams a response from the chat model while applying Layer 3 (AI response
      check) in the background.

    Args:
        request: The incoming FastAPI request, used for rate limiting.
        prompt: The user's input prompt.
        api_key: The validated API key from the dependency.

    Raises:
        HTTPException: If the prompt is too long.

    Returns:
        A streaming response with the chatbot's answer or a rejection message.
    """
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_413_PAYLOAD_TOO_LARGE,
            detail=f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters.",
        )

    prompt = prompt.strip()
    request_id = f"req-{id(prompt) % 10000:04d}"
    logger.info(f"[{request_id}] Prompt: {prompt[:100]}...")
    is_safe_event = asyncio.Event()
    is_safe_event.set()

    # Layer 1: Keyword block
    if any(kw in prompt.lower() for kw in FORBIDDEN_KEYWORDS):
        logger.warning(f"[{request_id}] L1 Guardrail: Forbidden keyword in prompt.")

        async def reject_l1() -> AsyncGenerator[str, None]:
            yield "\n\nขออภัยครับ ทาง SkyHigh Airlines ให้ข้อมูลเฉพาะบริการของเราเท่านั้น"

        return StreamingResponse(reject_l1(), media_type="text/plain; charset=utf-8")

    # Layer 2: Initial prompt check with AI Judge
    initial_check = asyncio.create_task(
        run_guardrail_judge(prompt, is_safe_event, f"{request_id}-prompt")
    )
    try:
        await asyncio.wait_for(initial_check, timeout=4.0)
        if not is_safe_event.is_set():
            logger.warning(f"[{request_id}] L2 Guardrail: Prompt judged as UNSAFE.")

            async def reject_l2() -> AsyncGenerator[str, None]:
                yield "\n\nขออภัยครับ เนื้อหานี้ไม่สอดคล้องกับนโยบายของเรา"

            return StreamingResponse(reject_l2(), media_type="text/plain; charset=utf-8")
    except asyncio.TimeoutError:
        logger.debug(f"[{request_id}] Initial judge timeout, proceeding with response.")
        initial_check.cancel()

    # Layer 3: Streaming response with ongoing checks
    async def stream_generator() -> AsyncGenerator[str, None]:
        """Generates response while running background checks."""
        client = ollama.AsyncClient()
        full_text_buffer: str = ""
        last_check_len: int = 0
        background_tasks: Set[asyncio.Task] = set()
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": "คุณคือพนักงานบริการลูกค้าของ SkyHigh Airlines...",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            async for part in await client.chat(
                model=CHAT_MODEL, messages=messages, stream=True
            ):
                if not is_safe_event.is_set():
                    logger.warning(f"[{request_id}] L3 Guardrail: Stream terminated.")
                    yield "\n\nขออภัยครับ การสนทนาไม่เป็นไปตามนโยบาย"
                    break

                if "message" in part and "content" in part["message"]:
                    token: str = part["message"]["content"]
                    full_text_buffer += token
                    yield token

                    if len(full_text_buffer) - last_check_len >= CHECK_INTERVAL:
                        window = full_text_buffer[-140:]
                        task = asyncio.create_task(
                            run_guardrail_judge(
                                window, is_safe_event, f"{request_id}-output"
                            )
                        )
                        background_tasks.add(task)
                        task.add_done_callback(background_tasks.discard)
                        last_check_len = len(full_text_buffer)
        except Exception as e:
            logger.error(f"[{request_id}] Stream error: {e}", exc_info=True)
            yield "\n\nขออภัยครับ ระบบมีปัญหาชั่วคราว"
        finally:
            for t in background_tasks:
                t.cancel()

    return StreamingResponse(
        stream_generator(), media_type="text/plain; charset=utf-8"
    )


if __name__ == "__main__":
    logger.info("✈️  SkyHigh Airlines Chatbot (Hardened API) started")
    logger.info(f"Chat model: {CHAT_MODEL}, Judge model: {JUDGE_MODEL}")
    logger.info(f"API Key authentication is ENABLED (Header: '{API_KEY_NAME}').")
    logger.info("Rate limiting is ENABLED (10 requests/minute/IP).")
    logger.info(f"CORS enabled for: {origins}")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")