# SkyHigh Airlines Chatbot (Strict Brand Guardrail PoC)

This project is a Proof-of-Concept (PoC) demonstrating a sophisticated, multi-layered asynchronous guardrail architecture for a customer service chatbot. It is built with FastAPI and uses a local LLM via Ollama.

The primary goal is to maintain strict brand safety and focus. The chatbot is designed for "SkyHigh Airlines" and must not discuss competitors, politics, or other off-topic subjects. It showcases a low-latency "optimistic streaming" approach, where responses are streamed to the user immediately while being validated in the background. If a violation is detected, a "circuit breaker" trips and terminates the stream gracefully.

## Core Features

- **Dual-LLM Architecture**: Uses a powerful model for generating chat responses (`CHAT_MODEL`) and a separate, lightweight model for validation (`JUDGE_MODEL`), ensuring high-quality answers without compromising on validation speed.
- **Multi-Layered Guardrails**: Implements a defense-in-depth strategy for content moderation:
    1.  **L1: Static Keyword Filter**: Instantly blocks user prompts containing forbidden keywords (e.g., competitor names).
    2.  **L2: AI Prompt Validation**: Before generating a response, an AI "judge" assesses the user's prompt for policy violations.
    3.  **L3: AI Response Validation**: The chatbot's own response is continuously monitored by the AI judge during streaming to catch any violations that may have been generated.
- **Asynchronous Circuit Breaker**: Utilizes `asyncio` for non-blocking I/O. An `asyncio.Event` acts as a circuit breaker. If any guardrail layer detects a violation, the event is triggered, immediately halting the stream to the user.
- **Performance Management**: A `Semaphore` is used to limit concurrent requests to the judge model, preventing it from being overloaded and ensuring stable performance.
- **Thai Language Focus**: The prompts, guardrail instructions, and user-facing messages are tailored for Thai language interactions.

## How It Works

1.  A user sends a prompt to the `/chat` endpoint.
2.  **Layer 1 Check**: The system scans the prompt for hardcoded `FORBIDDEN_KEYWORDS`. If found, a generic rejection message is returned immediately.
3.  **Layer 2 Check**: If the prompt passes Layer 1, it's sent to the `JUDGE_MODEL` for a quick AI-based safety check. If deemed `UNSAFE`, a rejection message is returned.
4.  **Streaming Response**: If the prompt is safe, the `CHAT_MODEL` begins generating a response, and the application starts streaming tokens to the user (Optimistic Streaming).
5.  **Layer 3 Check**: As the response is being generated, the text is accumulated. Periodically (based on `CHECK_INTERVAL`), chunks of the response are sent to the `JUDGE_MODEL` for ongoing validation in the background.
6.  **Circuit Breaker**: If at any point the `JUDGE_MODEL` returns `UNSAFE`, the `is_safe_event` is cleared, the main streaming loop breaks, and a final termination message is sent to the user.

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) running locally.
- The following Ollama models downloaded:
    - `gpt-oss:20b-cloud` (or another powerful chat model)
    - `qwen3:4b` (or another small, fast model for judging)

## Setup & Installation

1.  **Clone the repository or save `main.py`**.

2.  **Install Python dependencies**:
    ```bash
    pip install "fastapi[all]" uvicorn ollama
    ```

3.  **Download the Ollama models**:
    ```bash
    ollama pull gpt-oss:20b-cloud
    ollama pull qwen3:4b
    ```
    *Note: You can change the models used by editing the `CHAT_MODEL` and `JUDGE_MODEL` variables in `main.py`.*

## Running the Application

Execute the main file from your terminal:

```bash
python main.py
```

The server will start on `http://0.0.0.0:8000`.

## Usage & API Endpoint

The application exposes a single endpoint: `GET /chat`.

You can test it using `curl` or any HTTP client. The `prompt` should be a URL-encoded string.

### Example: Safe Prompt

This query is about the airline's own services and should stream a full response.

```bash
curl -G --data-urlencode "prompt=เที่ยวบินไปภูเก็ตราคาเท่าไหร่ครับ" http://127.0.0.1:8000/chat
```

### Example: Unsafe Prompt (Competitor)

This prompt should be blocked by the Layer 1 keyword filter.

```bash
curl -G --data-urlencode "prompt=AirAsia มีเที่ยวบินไปเชียงใหม่มั้ย" http://127.0.0.1:8000/chat
```

### Example: Unsafe Prompt (Will trigger AI Judge)

This prompt doesn't contain a keyword but will be flagged as `UNSAFE` by the Layer 2 AI judge.

```bash
curl -G --data-urlencode "prompt=คุณคิดยังไงกับรัฐบาลปัจจุบัน" http://127.0.0.1:8000/chat
```

## Configuration

You can modify the behavior of the guardrail by editing the constants at the top of `main.py`:

- `CHAT_MODEL`: The main model for generating responses.
- `JUDGE_MODEL`: The smaller model for validation.
- `CHECK_INTERVAL`: The number of characters to accumulate before running the response validation. A smaller number means more frequent checks but higher load.
- `JUDGE_TIMEOUT_SEC`: How long to wait for the judge model before failing open (assuming the content is safe).
- `FORBIDDEN_KEYWORDS`: The list of hardcoded keywords to block in user prompts.
- `GUARDRAIL_SYSTEM_PROMPT`: The master prompt that instructs the judge model on what is considered `SAFE` or `UNSAFE`.
