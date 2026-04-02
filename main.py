import os
import asyncio
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS for your GitHub Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIXED: Explicitly forcing 'v1' API to avoid v1beta 404 errors
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
    http_options={'api_version': 'v1'}
)

# Personality for SAZUG Assistant
SYSTEM_PROMPT = "You are the SAZUG AI Assistant. Use simple English and a friendly peer-to-peer tone."

@app.get("/")
async def root():
    return {"status": "SAZUG AI Backend is Live"}

@app.post("/")
async def chat_endpoint(
    text: str = Form(None),
    file_0: UploadFile = File(None),
    file_1: UploadFile = File(None),
    file_2: UploadFile = File(None)
):
    contents = []
    if text:
        contents.append(text)
    
    # Process files
    for upload_file in [file_0, file_1, file_2]:
        if upload_file:
            file_bytes = await upload_file.read()
            contents.append(
                types.Part.from_bytes(
                    data=file_bytes,
                    mime_type=upload_file.content_type
                )
            )

    if not contents:
        raise HTTPException(status_code=400, detail="Empty request")

    async def generate_stream():
        try:
            # FIXED: Using the specific stable model ID for 2026
            response = client.models.generate_content_stream(
                model='gemini-2.0-flash-001', 
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7
                )
            )
            
            for chunk in response:
                if chunk.text:
                    yield f"data: {chunk.text}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"Streaming Error: {e}")
            yield f"data: ⚠️ System is updating. Please try again in a moment.\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
