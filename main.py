"""
Main application file for the SkyHigh Airlines Chatbot.

This module sets up a FastAPI application with a multi-layered, asynchronous
guardrail system for an Ollama-powered chatbot. It includes rate limiting,
API key authentication, CORS, and input validation.
"""

import asyncio
import json
import logging
import os
import sys
from typing import AsyncGenerator, Dict, List, Set

import ollama
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Load environment variables from .env file at the beginning
load_dotenv()


# ==========================================
# Structured Logging Configuration
# ==========================================
class JsonFormatter(logging.Formatter):
    """Formats log records as JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            log_obj["exc_info"] = self.formatException(record.exc_info)
        
        # Add extra fields like 'violation'
        extra = record.__dict__.get('extra')
        if extra:
            log_obj.update(extra)
            
        return json.dumps(log_obj)


class ViolationFilter(logging.Filter):
    """Filters for log records that have the 'violation' flag."""
    def filter(self, record: logging.LogRecord) -> bool:
        return record.__dict__.get("extra", {}).get("violation", False)

# --- Root Logger Setup (Console) ---
json_formatter = JsonFormatter()
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(json_formatter)

# --- Violation Logger Setup (File) ---
violation_handler = logging.FileHandler("violations.log", mode="a")
violation_handler.setFormatter(json_formatter)
violation_handler.addFilter(ViolationFilter())


# Configure the root logger and add handlers
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), handlers=[stream_handler])
logger: logging.Logger = logging.getLogger("skyhigh-airlines-chatbot")

# Add the dedicated violation handler to the app's logger
logging.getLogger("skyhigh-airlines-chatbot").addHandler(violation_handler)


# ==========================================
# Application Configuration
# ==========================================
API_KEY: str = os.getenv("API_KEY", "skyhigh-secret-key-for-dev")
API_KEY_NAME: str = "X-API-KEY"
MAX_PROMPT_LENGTH: int = 1000

limiter: Limiter = Limiter(key_func=get_remote_address)
origins: List[str] = [
    "http://localhost", "http://localhost:3000", "http://localhost:8080"
]

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
# API Key Authentication & Other Dependencies
# ==========================================
api_key_header: APIKeyHeader = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


async def get_api_key(api_key_header_value: str = Security(api_key_header)) -> str:
    """Validates the API key provided in the request header."""
    if api_key_header_value == API_KEY:
        return api_key_header_value
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API Key"
    )


# ==========================================
# Models, Prompts, and Guardrail Logic
# ==========================================
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-oss:20b-cloud")
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "qwen3:4b")
CHECK_INTERVAL: int = 40
JUDGE_TIMEOUT_SEC: float = 5.0
MAX_CONCURRENT_JUDGES: int = 1
judge_semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT_JUDGES)

FORBIDDEN_KEYWORDS: List[str] = [
    "airasia", "แอร์เอเชีย", "nok air", "นกแอร์", "lion air", "ไลอ้อนแอร์",
    "vietjet", "qatar airways", "emirates", "singapore airlines", "การบินไทย",
]

GUARDRAIL_SYSTEM_PROMPT: str = "..." # Unchanged


async def run_guardrail_judge(
    text: str, is_safe_event: asyncio.Event, request_id: str = ""
) -> None:
    """Runs a check on the given text using the JUDGE_MODEL."""
    if not text.strip() or judge_semaphore.locked():
        return

    async with judge_semaphore:
        try:
            async with asyncio.timeout(JUDGE_TIMEOUT_SEC):
                prompt = GUARDRAIL_SYSTEM_PROMPT.format(text=text)
                client = ollama.AsyncClient()
                resp = await client.generate(model=JUDGE_MODEL, prompt=prompt, stream=False)
                if "UNSAFE" in resp["response"].strip().upper():
                    logger.warning(
                        f"[{request_id}] UNSAFE detected",
                        extra={"violation": True, "requestId": request_id, "text": text[-60:]}
                    )
                    is_safe_event.clear()
        except asyncio.TimeoutError:
            logger.warning(f"[{request_id}] Judge timeout → fail-open.")
        except Exception as e:
            logger.error(f"[{request_id}] Judge error: {e}", exc_info=True)


# ==========================================
# API Endpoints
# ==========================================
@app.get("/health", tags=["Monitoring"])
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint to confirm the API is running.
    Does not require authentication.
    """
    return {"status": "ok"}


@app.get("/chat", tags=["Chat"])
@limiter.limit("10/minute")
async def chat_endpoint(
    request: Request, prompt: str, api_key: str = Depends(get_api_key)
) -> StreamingResponse:
    """Main chat endpoint with multi-layered security and guardrails."""
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_413_PAYLOAD_TOO_LARGE,
            detail=f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters.",
        )

    prompt = prompt.strip()
    request_id = f"req-{id(prompt) % 10000:04d}"
    logger.info(f"[{request_id}] New prompt received.", extra={"requestId": request_id, "prompt_start": prompt[:100]})
    
    is_safe_event = asyncio.Event()
    is_safe_event.set()

    if any(kw in prompt.lower() for kw in FORBIDDEN_KEYWORDS):
        logger.warning(
            f"[{request_id}] L1 Guardrail: Forbidden keyword in prompt.",
            extra={"violation": True, "requestId": request_id, "keyword": kw}
        )
        async def reject_l1() -> AsyncGenerator[str, None]:
            yield "ขออภัยค่ะ ให้ข้อมูลเฉพาะบริการของเราเท่านั้น"
        return StreamingResponse(reject_l1(), media_type="text/plain; charset=utf-8")

    initial_check = asyncio.create_task(
        run_guardrail_judge(prompt, is_safe_event, f"{request_id}-prompt")
    )
    try:
        await asyncio.wait_for(initial_check, timeout=4.0)
        if not is_safe_event.is_set():
            logger.warning(
                f"[{request_id}] L2 Guardrail: Prompt judged as UNSAFE.",
                extra={"violation": True, "requestId": request_id}
            )
            async def reject_l2() -> AsyncGenerator[str, None]:
                yield "ขออภัยค่ะ เนื้อหานี้ไม่สอดคล้องกับนโยบายของเรา"
            return StreamingResponse(reject_l2(), media_type="text/plain; charset=utf-8")
    except asyncio.TimeoutError:
        logger.debug(f"[{request_id}] Initial judge timeout, proceeding.")
        initial_check.cancel()

    async def stream_generator() -> AsyncGenerator[str, None]:
        # ... (streaming logic is the same)
        pass # Placeholder for brevity

    return StreamingResponse(stream_generator(), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    logger.info("✈️  SkyHigh Airlines Chatbot (Hardened API) starting up...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")