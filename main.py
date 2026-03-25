import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="""
You are SAZUG AI assistant.

Help students with:
- School-related questions
- Exams and past questions

Always respond clearly, simply, and professionally.
"""
)

@app.post("/api/chat")
async def chat(request: Request):
    try:
        form = await request.form()
        text = form.get("text", "")

        content = [text] if text else []

        for key in form:
            if key.startswith("file_"):
                file = form[key]
                file_data = await file.read()

                if file.content_type.startswith("image/"):
                    content.append({
                        "mime_type": file.content_type,
                        "data": file_data
                    })

        if not content:
            return {"response": "Please send a message."}

        response = model.generate_content(content)

        return {"response": response.text}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)