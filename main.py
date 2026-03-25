import os
import logging
from fastapi import FastAPI, Form, UploadFile, File, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio
import io
from typing import List

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for your frontend (adjust allowed origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                     # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Gemini API Key from environment variables
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")
genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-1.5-pro"               # Multimodal model

@app.get("/")
async def root():
    return {"message": "SAZUG AI Backend is running. Use POST with text and files."}

@app.post("/")
async def chat_endpoint(request: Request):
    """
    Accepts multipart/form-data with:
    - text: the user's message
    - file_0, file_1, ...: uploaded files (images, PDFs, videos)

    Returns an SSE stream of the model's response.
    """
    try:
        form = await request.form()
        text = form.get("text", "")

        # Collect all uploaded files (keys like "file_0", "file_1", ...)
        files = []
        for key, value in form.items():
            if key.startswith("file_") and isinstance(value, UploadFile):
                files.append(value)

        # Build contents for Gemini
        contents = []

        # Add text part if present
        if text:
            contents.append(text)           # Gemini accepts plain text as a part

        # Process each file
        for file in files:
            mime_type = file.content_type or "application/octet-stream"
            try:
                file_bytes = await file.read()
                # Create a Part from raw bytes
                from google.generativeai.types import Part
                part = Part.from_data(data=file_bytes, mime_type=mime_type)
                contents.append(part)
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                continue

        if not contents:
            raise HTTPException(status_code=400, detail="No content provided")

        model = genai.GenerativeModel(MODEL_NAME)

        # Streaming generator
        async def stream_generator():
            try:
                response = await model.generate_content_async(contents, stream=True)
                async for chunk in response:
                    if chunk.text:
                        yield f"data: {chunk.text}\n\n"
                # Optional: send a [DONE] marker (ignored by frontend)
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                yield f"data: Error: {str(e)}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.exception("Error in chat endpoint")
        raise HTTPException(status_code=500, detail=str(e))