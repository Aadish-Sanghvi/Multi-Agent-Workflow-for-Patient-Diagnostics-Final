from openai import OpenAI
from config import gemma_api, prompt_normal, prompt_with_context

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
    # Open the image file in binary mode
    with open(r"C:\Users\abhin\Desktop\Multi-Agent-Workflow-for-Patient-Diagnostics\utils\image.png", "rb") as image_file:
        
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
                        {"type": "image_file", r"image_file": "C:\Users\abhin\Desktop\Multi-Agent-Workflow-for-Patient-Diagnostics\utils\__pycache__\image.png"}  # <--- Send file directly
                    ]
                }
            ]
        )

    print(completion.choices[0].message.content)

except Exception as e:
    print(f"Error: {e}")
