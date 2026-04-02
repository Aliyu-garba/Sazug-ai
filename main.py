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

# Enable CORS for GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Define the Personality/System Instruction
SYSTEM_PROMPT = """
You are the SAZUG AI Assistant, a friendly and helpful academic companion for students at Sa'adu Zungur University (SAZUG) in Bauchi State, Nigeria. 
Your tone is peer-to-peer—like a knowledgeable friend explaining things the night before an exam. 
Use simple English and clear, step-by-step definitions. 
If asked about past questions, school fees, or the creator, provide helpful guidance based on the university's context.
Always be encouraging and supportive!
"""

@app.get("/")
async def root():
    return {"status": "SAZUG AI is Live"}

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
    
    # Process up to 3 files from the frontend
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
            # We add the system_instruction here
            response = client.models.generate_content_stream(
                model='gemini-1.5-flash',
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7 # Makes the tone more natural
                )
            )
            
            for chunk in response:
                if chunk.text:
                    # Matches your frontend line: if (line.startsWith('data: '))
                    yield f"data: {chunk.text}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"Error: {e}")
            yield f"data: Error: The AI service is currently busy. Please try again in a moment.\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
