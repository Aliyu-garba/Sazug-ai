import os
import logging
import asyncio
from typing import List

from fastapi import FastAPI, Request, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Using the modern Google GenAI SDK to avoid import errors
from google import genai
from google.genai import types
from dotenv import load_dotenv

# -------------------- LOAD ENV --------------------
load_dotenv()

# -------------------- LOGGING --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- FASTAPI APP --------------------
app = FastAPI()

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- API KEY & CLIENT --------------------
API_KEY = os.getenv("GEMINI_API_KEY")

# Using 'gemini-1.5-flash' for better stability on Free Tier
# Change to 'gemini-1.5-pro' if you require higher reasoning power
MODEL_NAME = "gemini-1.5-flash" 

if not API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY is NOT set. Backend will not work!")

# Initialize the modern client
client = genai.Client(api_key=API_KEY)

# -------------------- ROOT ROUTE --------------------
@app.get("/")
async def root():
    return {"message": "✅ SAZUG AI Backend is running!"}

# -------------------- CHAT ENDPOINT --------------------
@app.post("/")
async def chat_endpoint(request: Request):
    try:
        form = await request.form()
        user_text = form.get("text", "")

        # Collect uploaded files
        files: List[UploadFile] = []
        for key, value in form.items():
            if key.startswith("file_") and isinstance(value, UploadFile):
                files.append(value)

        # Prepare content list for Gemini
        contents = []
        
        # Add text if present
        if user_text:
            contents.append(user_text)

        # Add files as parts
        for file in files:
            try:
                file_bytes = await file.read()
                part = types.Part.from_bytes(
                    data=file_bytes,
                    mime_type=file.content_type or "application/octet-stream"
                )
                contents.append(part)
            except Exception as e:
                logger.error(f"❌ File processing error ({file.filename}): {e}")

        if not contents:
            raise HTTPException(status_code=400, detail="No content provided")

        # -------------------- STREAMING GENERATOR --------------------
        async def stream_generator():
            try:
                # Use the asynchronous models generator with explicit model string
                async for chunk in await client.aio.models.generate_content_stream(
                    model=MODEL_NAME,
                    contents=contents
                ):
                    if chunk.text:
                        yield f"data: {chunk.text}\n\n"
                        await asyncio.sleep(0.01)

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
