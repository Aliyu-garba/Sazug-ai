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

# Enable CORS so your GitHub Pages frontend can talk to Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Gemini Client
# Make sure GEMINI_API_KEY is set in your Render Environment Variables
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@app.get("/")
async def root():
    return {"status": "SAZUG AI Backend is running"}

@app.post("/")
async def chat_endpoint(
    text: str = Form(None),
    file_0: UploadFile = File(None),
    file_1: UploadFile = File(None),
    file_2: UploadFile = File(None)
):
    # Collect provided text and files
    contents = []
    if text:
        contents.append(text)
    
    for upload_file in [file_0, file_1, file_2]:
        if upload_file:
            try:
                file_bytes = await upload_file.read()
                # Use types.Part.from_bytes for the new GenAI SDK
                contents.append(
                    types.Part.from_bytes(
                        data=file_bytes,
                        mime_type=upload_file.content_type
                    )
                )
            except Exception as e:
                print(f"Error processing file: {e}")

    if not contents:
        raise HTTPException(status_code=400, detail="No text or files provided")

    async def generate_stream():
        try:
            # We use 'gemini-1.5-flash' (latest stable version)
            # This matches the requirement of your multimodal frontend
            response = client.models.generate_content_stream(
                model='gemini-1.5-flash',
                contents=contents
            )
            
            for chunk in response:
                if chunk.text:
                    # Format as Server-Sent Events (SSE) for your frontend
                    yield f"data: {chunk.text}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Render provides a PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
