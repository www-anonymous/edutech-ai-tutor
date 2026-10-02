import os
import json
import base64
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
from google import genai
from google.genai import types

from pathlib import Path
# Load local .env only if the physical file exists
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

app = FastAPI(title="EduTech AI Tutor")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Accept either variable name from Render or local environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    print("\n❌ CRITICAL: No GEMINI_API_KEY found in environment or .env!\n")
    client = None
else:
    print(f"\n✅ GEMINI_API_KEY loaded: {GEMINI_API_KEY[:6]}... (Length: {len(GEMINI_API_KEY)})\n")
    client = genai.Client(api_key=GEMINI_API_KEY)
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    student_level: str
    mode: str
    language: Optional[str] = "en"
    image_base64: Optional[str] = None
    history: Optional[List[Message]] = []

def build_system_instruction(student_level: str, mode: str, language: str) -> str:
    level_guidelines = {
        "10th": "Target: 10th Class (Board Exams / NCERT). Keep explanations foundational, clear, and relatable.",
        "inter": "Target: Intermediate (+2 / 11th-12th / JEE / NEET). Cover derivations, conceptual shortcuts, and board/entrance patterns.",
        "degree": "Target: Degree / Undergraduate. Provide rigorous academic reasoning, proofs, and technical precision."
    }

    pedagogy = (
        "PEDAGOGY MODE: SOCRATIC TUTOR. Never output the complete answer at once. Give Step 1 or an intuitive hint, then ask the student an interactive question."
        if mode == "socratic" else
        "PEDAGOGY MODE: DIRECT BREAKDOWN. Provide a full verified step-by-step solution with definitions and formulas."
    )

    lang_rules = {
        "hi": "PRIMARY LANGUAGE: Respond in Hindi (using clean Devanagari script, or Hinglish if the user asks in Hinglish). Keep standard math equations, formulas, and units in English/Latin script.",
        "te": "PRIMARY LANGUAGE: Respond in Telugu (using Telugu script). Keep standard math equations, formulas, and units in English/Latin script.",
        "en": "PRIMARY LANGUAGE: Respond in clear, accessible English."
    }

    return f"""You are an elite, patient EduTech AI tutor.
{level_guidelines.get(student_level, level_guidelines['10th'])}
{pedagogy}
{lang_rules.get(language, lang_rules['en'])}

FORMATTING RULES:
1. ALWAYS format math equations with LaTeX: $...$ for inline, and $$...$$ for standalone display equations.
2. Structure answers with clean markdown headings and bullet points where helpful.
"""

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return FileResponse("static/index.html")

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    if not client:
        async def err_stream():
            yield f"data: {json.dumps({'error': '⚠️ GEMINI_API_KEY is missing from .env file.'})}\n\n"
        return StreamingResponse(err_stream(), media_type="text/event-stream")

    sys_instruction = build_system_instruction(req.student_level, req.mode, req.language)
    
    contents = []
    if req.history:
        for msg in req.history[-6:]:
            contents.append(f"{msg.role.capitalize()}: {msg.content}")

    if req.image_base64:
        try:
            data = req.image_base64.split(",")[-1] if "," in req.image_base64 else req.image_base64
            img_bytes = base64.b64decode(data)
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    user_text = req.message if req.message else "Explain or solve this step-by-step."
    contents.append(f"Student: {user_text}")

    async def sse_generator():
        # Models supported by Google AI Studio
        candidate_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
        stream = None

        for model_name in candidate_models:
            try:
                stream = await client.aio.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_instruction,
                        temperature=0.4
                    )
                )
                break
            except Exception:
                continue

        if not stream:
            yield f"data: {json.dumps({'error': 'All models are currently busy or unavailable. Please retry in a few seconds.'})}\n\n"
            return

        try:
            async for chunk in stream:
                if await request.is_disconnected():
                    break
                if chunk.text:
                    payload = json.dumps({"token": chunk.text})
                    yield f"data: {payload}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)