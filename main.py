import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import uvicorn
from typing import Optional

app = FastAPI()

# Enable CORS so your website can talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="You are SAZUG AI assistant. Help students with school questions, exams, and data guidance. Be clear and professional."
)

@app.post("/chat")
async def chat(message: str = Form(...), image: Optional[UploadFile] = File(None)):
    try:
        content = []
        if message:
            content.append(message)
        
        if image:
            image_data = await image.read()
            content.append({
                "mime_type": image.content_type,
                "data": image_data
            })

        if not content:
            return {"reply": "Please provide a message or an image."}

        # Generate response from Gemini
        response = model.generate_content(content)
        
        # We use 'reply' to match what your frontend's data.reply expects
        return {"reply": response.text}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": "The AI is having trouble thinking. Check your API key or connection."}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
