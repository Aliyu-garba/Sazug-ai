import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import uvicorn

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment variable (SAFE)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Improved system instruction
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="""
You are SAZUG AI assistant.

Help students with:
- School-related questions
- Exams and past questions
- Clear and simple explanations

Always:
- Respond clearly and professionally
- Use simple language (easy to understand)
- Be helpful and direct
"""
)

@app.post("/chat")
async def chat(message: str = Form(...), image: UploadFile = File(None)):
    try:
        content = [message]

        # If image is included
        if image:
            image_data = await image.read()
            content.append({
                "mime_type": image.content_type,
                "data": image_data
            })

        # Generate AI response
        response = model.generate_content(content)

        return {"reply": response.text}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)