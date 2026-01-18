import asyncio
import logging
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import ollama

# ==========================================
# Logging & Configuration
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skyhigh-airlines-chatbot")

app = FastAPI(title="SkyHigh Airlines Chatbot (Strict Brand Guardrail)")

# โมเดลแนะนำ (ปรับตามเครื่องคุณ)
CHAT_MODEL  = "gpt-oss:20b-cloud"          # ตัวตอบหลัก (ควรใช้ตัวใหญ่หน่อย)
JUDGE_MODEL = "qwen3:4b"           # ตัวตรวจ (1.5b ขึ้นไปจะแม่นกว่ามาก)

CHECK_INTERVAL    = 40
JUDGE_TIMEOUT_SEC = 5.0
MAX_CONCURRENT_JUDGES = 1

judge_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JUDGES)

# ชั้นดิบ – บล็อกทันทีถ้าพบคำเหล่านี้ใน prompt
FORBIDDEN_KEYWORDS = [
    "airasia", "แอร์เอเชีย",
    "nok air", "นกแอร์",
    "lion air", "ไลอ้อนแอร์", "thai lion",
    "vietjet",
    "qatar airways", "กาตาร์แอร์เวย์", "qatar", "กาตาร์", "qatar airways",
    "emirates", "เอมิเรตส์",
    "singapore airlines", "การบินไทย", "thaiairways",
    # เพิ่มได้ตามต้องการ เช่น "bangkok airways", "เจแปนแอร์ไลน์" ฯลฯ
]

# ==========================================
# Guardrail Prompt – เข้มงวดกับสายการบินอื่นทั้งหมด
# ==========================================
GUARDRAIL_SYSTEM_PROMPT = """\
คุณคือนักตรวจสอบเนื้อหาที่เข้มงวดมากสำหรับแชทบอทบริการลูกค้าของสายการบิน SkyHigh Airlines
ตอบได้แค่คำเดียว: SAFE หรือ UNSAFE

ถือว่า UNSAFE ทันที หากข้อความมี ANY อย่างใดอย่างหนึ่ง ดังนี้ (แม้จะเป็นคำเดียว คำถาม การบ่น หรือการเปรียบเทียบ):
- การเมือง, นายกรัฐมนตรี, รัฐบาล, ประท้วง, การเลือกตั้ง, มาตรา 112
- ชื่อสายการบินใด ๆ ที่ไม่ใช่ SkyHigh Airlines เช่น Qatar Airways, AirAsia, Nok Air, Lion Air, VietJet, Emirates, Singapore Airlines, การบินไทย ฯลฯ
- ศาสนา, พระ, พระเจ้า, วัด, มัสยิด, เจ้าอาวาส (ยกเว้นคำทั่วไปเช่น สงกรานต์ ปีใหม่)

ตัวอย่าง:
"Qatar Airways ดีกว่า SkyHigh มั้ย"     → UNSAFE
"จอง Qatar ไปลอนดอนราคาเท่าไหร่"       → UNSAFE
"แอร์เอเชียถูกกว่า"                     → UNSAFE
"บินไปภูเก็ตกับ SkyHigh เท่าไหร่ครับ"    → SAFE
"ผมอยากบ่นเรื่องรัฐบาล"                 → UNSAFE

ข้อความที่ต้องตรวจ: "{text}"

ตอบแค่ SAFE หรือ UNSAFE เท่านั้น ห้ามอธิบายเพิ่ม
"""

# ==========================================
# ฟังก์ชันตรวจ guardrail
# ==========================================
async def run_guardrail_judge(
    text: str,
    is_safe_event: asyncio.Event,
    request_id: str = ""
) -> None:
    if not text.strip():
        return

    if judge_semaphore.locked():
        logger.debug(f"[{request_id}] Judge busy → skip")
        return

    async with judge_semaphore:
        try:
            async with asyncio.timeout(JUDGE_TIMEOUT_SEC):
                prompt = GUARDRAIL_SYSTEM_PROMPT.format(text=text)
                client = ollama.AsyncClient()
                resp = await client.generate(
                    model=JUDGE_MODEL,
                    prompt=prompt,
                    options={"temperature": 0.0},
                    stream=False
                )
                judgement = resp['response'].strip().upper()

                if "UNSAFE" in judgement:
                    logger.warning(f"[{request_id}] UNSAFE detected: ...{text[-60:]}")
                    is_safe_event.clear()

        except asyncio.TimeoutError:
            logger.warning(f"[{request_id}] Judge timeout → fail-open")
        except Exception as e:
            logger.error(f"[{request_id}] Judge error: {e}", exc_info=True)


# ==========================================
# Endpoint หลัก
# ==========================================
@app.get("/chat")
async def chat_endpoint(prompt: str):
    request_id = f"req-{id(prompt) % 10000:04d}"
    logger.info(f"[{request_id}] Prompt: {prompt[:100]}...")

    is_safe_event = asyncio.Event()
    is_safe_event.set()

    prompt_lower = prompt.lower()

    # ชั้น 1: Keyword block ด่วน (เร็วที่สุด)
    if any(kw in prompt_lower for kw in FORBIDDEN_KEYWORDS):
        logger.warning(f"[{request_id}] Forbidden keyword in prompt")
        async def reject():
            yield "\n\nขออภัยครับ ทาง SkyHigh Airlines สามารถให้ข้อมูลและช่วยเหลือเฉพาะเรื่องบริการของเราเท่านั้น "
            yield "กรุณาถามเกี่ยวกับการจองตั๋ว เที่ยวบิน โปรโมชั่น หรือบริการของ SkyHigh Airlines ได้เลยนะครับ ยินดีช่วยเหลือครับ!"
        return StreamingResponse(reject(), media_type="text/plain; charset=utf-8")

    # ชั้น 2: ตรวจ prompt ด้วย judge ก่อนเริ่มตอบ
    initial_check = asyncio.create_task(
        run_guardrail_judge(prompt, is_safe_event, request_id + "-prompt")
    )

    try:
        await asyncio.wait_for(initial_check, timeout=4.0)
        if not is_safe_event.is_set():
            async def immediate_reject():
                yield "\n\nขออภัยครับ เนื้อหานี้ไม่สอดคล้องกับนโยบายการให้บริการของ SkyHigh Airlines "
                yield "กรุณาถามเรื่องการเดินทาง การจอง หรือบริการของเรานะครับ ยินดีให้ความช่วยเหลือครับ!"
            return StreamingResponse(immediate_reject(), media_type="text/plain; charset=utf-8")
    except asyncio.TimeoutError:
        logger.debug(f"[{request_id}] Initial judge timeout → proceed")
        initial_check.cancel()

    # ── ถ้าผ่านทั้งสองชั้น → เริ่มตอบปกติ ──
    background_tasks = set()

    def done_callback(task):
        background_tasks.discard(task)

    async def stream_generator() -> AsyncGenerator[str, None]:
        client = ollama.AsyncClient()
        full_text_buffer = ""
        last_check_len = 0

        messages = [
            {
                "role": "system",
                "content": """
คุณคือพนักงานบริการลูกค้าของ SkyHigh Airlines เท่านั้น
- ตอบเฉพาะเรื่องของ SkyHigh Airlines: การจองตั๋ว, เที่ยวบิน, ตารางบิน, บริการบนเครื่อง, สะสมไมล์, โปรโมชั่น, นโยบายกระเป๋าเดินทาง ฯลฯ
- ห้ามกล่าวถึง ให้ข้อมูล เปรียบเทียบ หรือตอบคำถามเกี่ยวกับสายการบินอื่นใดทั้งสิ้น แม้ผู้ใช้จะถามมาโดยตรง
- หากผู้ใช้ถามถึงสายการบินอื่น ให้ตอบสุภาพว่า "ขออภัยครับ ทางเราสามารถให้ข้อมูลเฉพาะของ SkyHigh Airlines ได้เท่านั้น กรุณาถามเรื่องของเรานะครับ"
- ตอบเป็นภาษาไทย สุภาพ เป็นกันเอง กระชับ และเป็นประโยชน์
                """
            },
            {"role": "user", "content": prompt}
        ]

        try:
            async for part in await client.chat(
                model=CHAT_MODEL,
                messages=messages,
                stream=True,
                options={"temperature": 0.7}
            ):
                if not is_safe_event.is_set():
                    yield "\n\nขออภัยครับ เนื้อหาที่กำลังสนทนาไม่เป็นไปตามนโยบายของสายการบิน "
                    yield "กรุณาถามเรื่องการเดินทางหรือบริการของ SkyHigh Airlines เพิ่มเติมได้เลยนะครับ"
                    break

                if 'message' in part and 'content' in part['message']:
                    token = part['message']['content']
                    full_text_buffer += token
                    yield token

                    current_len = len(full_text_buffer)
                    if current_len - last_check_len >= CHECK_INTERVAL:
                        window = full_text_buffer[-140:]
                        task = asyncio.create_task(
                            run_guardrail_judge(window, is_safe_event, request_id + "-output")
                        )
                        task.add_done_callback(done_callback)
                        background_tasks.add(task)
                        last_check_len = current_len

        except Exception as e:
            logger.error(f"[{request_id}] Stream error: {e}", exc_info=True)
            yield "\n\nขออภัยครับ ระบบกำลังมีปัญหาชั่วคราว กรุณาลองใหม่อีกครั้งนะครับ"

        finally:
            for t in list(background_tasks):
                t.cancel()
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)

    return StreamingResponse(
        stream_generator(),
        media_type="text/plain; charset=utf-8"
    )


if __name__ == "__main__":
    logger.info("✈️  SkyHigh Airlines Chatbot (Strict Brand Focus) started")
    logger.info(f"Chat model : {CHAT_MODEL}")
    logger.info(f"Judge model: {JUDGE_MODEL}")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")