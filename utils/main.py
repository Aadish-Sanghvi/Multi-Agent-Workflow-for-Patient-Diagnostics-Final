from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from openai import OpenAI
from config import gemma_api, prompt_normal, prompt_with_context
import base64
import io

app = FastAPI(title="Medical Image Analysis API")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=gemma_api
)

@app.post("/analyze_base64")
async def analyze_base64_image(
    image_base64: str = Form(...),       # base64 string
    conversation: str = Form(default="")
):
    try:
        # Prepare prompt
        prompt = prompt_normal if not conversation.strip() else prompt_with_context.format(conversation=conversation)

        # Convert base64 back to file-like object
        header, encoded = image_base64.split(",", 1) if "," in image_base64 else ("", image_base64)
        image_bytes = io.BytesIO(base64.b64decode(encoded))

        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>",
                "X-Title": "<YOUR_SITE_NAME>"
            },
            model="google/gemma-3-27b-it:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_file", "image_file": image_bytes}
                    ]
                }
            ]
        )

        result = completion.choices[0].message.content
        return JSONResponse(content={"analysis": result})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
