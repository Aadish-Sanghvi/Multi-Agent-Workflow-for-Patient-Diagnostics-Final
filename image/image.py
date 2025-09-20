from openai import OpenAI
from config import gemma_api, prompt_normal, prompt_with_context
import base64

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=gemma_api
)

# Conversation is optional
conversation = ""   # <-- keep empty or put conversation string

# Choose prompt depending on conversation
if not conversation.strip():
    prompt = prompt_normal
else:
    prompt = prompt_with_context.format(conversation=conversation)

try:
    with open(r"C:\Users\abhin\Desktop\Multi-Agent-Workflow-for-Patient-Diagnostics\utils\image.png", "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
    image_base64 = f"data:image/png;base64,{b64_data}"

    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "<YOUR_SITE_URL>",   # Optional
            "X-Title": "<YOUR_SITE_NAME>",       # Optional
        },
        model="google/gemma-3-27b-it:free",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_base64", "image_base64": image_base64}
                ]
            }
        ]
    )
    print(completion.choices[0].message.content)

except Exception as e:
    print(f"Error: {e}")
