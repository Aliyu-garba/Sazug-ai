import os
import logging
import asyncio
from typing import List

from fastapi import FastAPI, Request, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import google.generativeai as genai
from dotenv import load_dotenv

# -------------------- LOAD ENV --------------------
load_dotenv()

# -------------------- LOGGING --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- FASTAPI APP --------------------
app = FastAPI()

# -------------------- CORS (ALLOW FRONTEND) --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ⚠️ Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- API KEY --------------------
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY is NOT set. Backend will not work!")

genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-1.5-pro"

# -------------------- ROOT ROUTE --------------------
@app.get("/")
async def root():
    return {"message": "✅ SAZUG AI Backend is running!"}

# -------------------- CHAT ENDPOINT --------------------
@app.post("/")
async def chat_endpoint(request: Request):
    try:
        form = await request.form()

        text = form.get("text", "")

        # Collect uploaded files
        files: List[UploadFile] = []
        for key, value in form.items():
            if key.startswith("file_") and isinstance(value, UploadFile):
                files.append(value)

        contents = []

        # Add text
        if text:
            contents.append(text)

        # Add files
        from google.generativeai.types import Part

        for file in files:
            try:
                file_bytes = await file.read()
                part = Part.from_data(
                    data=file_bytes,
                    mime_type=file.content_type or "application/octet-stream"
                )
                contents.append(part)
            except Exception as e:
                logger.error(f"❌ File error ({file.filename}): {e}")

        if not contents:
            raise HTTPException(status_code=400, detail="No content provided")

        model = genai.GenerativeModel(MODEL_NAME)

        # -------------------- STREAMING FUNCTION --------------------
        async def stream_generator():
            try:
                response = await model.generate_content_async(
                    contents,
                    stream=True
                )

                async for chunk in response:
                    if hasattr(chunk, "text") and chunk.text:
                        yield f"data: {chunk.text}\n\n"
                        await asyncio.sleep(0.01)  # 🔥 prevents buffering

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"❌ Streaming error: {e}")
                yield f"data: Error: {str(e)}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.exception("❌ Chat endpoint error")
        raise HTTPException(status_code=500, detail=str(e))