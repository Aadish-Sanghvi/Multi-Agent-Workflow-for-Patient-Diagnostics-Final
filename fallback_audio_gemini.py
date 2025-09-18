import json
import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import google.generativeai as genai
from config import *


# Silence GRPC warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiClient:
    def __init__(self, api_key: str, model_name: str, system_prompt: str):
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.client = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(
                self.model_name,
                system_instruction=self.system_prompt
            )
            logger.info(f"Gemini model '{self.model_name}' initialized with system prompt!")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
            self.client = None

    def generate_from_audio(self, audio_path: str, text_prompt: str) -> Optional[str]:
        """Send audio file + text instruction to Gemini."""
        if not self.client:
            logger.error("Gemini model not available.")
            return None

        try:
            # Determine MIME type based on file extension
            ext = os.path.splitext(audio_path)[1].lower()
            if ext == ".mp3":
                mime = "audio/mp3"
            elif ext in [".wav"]:
                mime = "audio/wav"
            elif ext in [".m4a"]:
                mime = "audio/m4a"
            else:
                mime = "audio/wav"  # fallback

            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            response = self.client.generate_content([
                {"mime_type": mime, "data": audio_bytes},
                {"text": text_prompt}
            ])
            return response.text.strip()

        except Exception as e:
            logger.error(f"Gemini audio generation error: {str(e)}")
            return None


class RoleSeparator:
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client

    def separate_roles(self, audio_path: str, patient_id: str = "default") -> Dict[str, Any]:
        """Takes audio file path, runs through Gemini, parses response as JSON."""
        if not audio_path or not os.path.exists(audio_path):
            logger.error("Audio file not found.")
            return {"error": "Audio file not found."}

        result_text = self.gemini_client.generate_from_audio(
            audio_path,
            "Transcribe the audio and convert it into structured JSON chat format "
            "between doctor and patient with proper role separation."
        )

        if not result_text:
            return {"error": "Gemini model not available or failed to generate content."}

        result_json = self._parse_json(result_text)
        if not result_json:
            logger.error("Failed to parse Gemini response.")
            return {"error": "Failed to parse Gemini response."}

        timestamp = datetime.now(timezone.utc).isoformat()
        safe_timestamp = timestamp.replace(":", "_")

        if isinstance(result_json, list):
            output = {
                "conversation": result_json,
                "patient_id": patient_id,
                "timestamp": timestamp,
            }
        else:
            output = result_json
            output["patient_id"] = patient_id
            output["timestamp"] = timestamp

    # Removed saving output to file as per user request
        return output

    def _parse_json(self, result_text: str) -> Optional[Dict[str, Any]]:
        start_idx = result_text.find("[")
        end_idx = result_text.rfind("]") + 1
        if start_idx != -1 and end_idx != 0:
            try:
                return json.loads(result_text[start_idx:end_idx])
            except Exception as e:
                logger.error(f"JSON parsing error: {str(e)}")
                return None
        return None

    def _save_output(self, result_json: Dict[str, Any], patient_id: str, safe_timestamp: str):
        output_path = f"role_separation_{patient_id}_{safe_timestamp}.json"
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result_json, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved role separation output to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save output: {str(e)}")


def main():
    import time
    audio_path = r"/Users/abhinavgupta/Desktop/projects/New Recording.mp3"

    gemini_client = GeminiClient(
        api_key=gemini_api_key,
        model_name=gemini_model_name,
        system_prompt="""
            You are a multilingual expert who understands multiple languages.  
    You will receive audio files in different languages.  

    1. First, detect the language of the audio.  
    2. If the audio is in Hindi or Hinglish, convert it into **Hinglish** (preferred).  
    3. For all other languages, convert the audio into **English** (or Hindi if it improves clarity).  
    4. Always ensure the output is in Hinglish or English.  
    5. If the audio is in Hindi or a mix of Hindi and English, then convert it into Hinglish only.  
    6. If the audio is in a language you do not understand, respond with "Language not supported".  
    You are also an expert at converting **medical consultation audio** into an accurate **doctor–patient chat format** with clear role separation (Doctor vs Patient).  

    ⚡ Pay special attention to:  
    - Preserving medical terminology  
    - Maintaining the correct context  
    - Ensuring clarity in the conversation  

    The **final output** must always be in **Hinglish or English**, closely reflecting the original audio content.
"""
    )

    role_separator = RoleSeparator(gemini_client)

    start_time = time.time()
    result = role_separator.separate_roles(audio_path, patient_id="PAT-101")
    end_time = time.time()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nProcess time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
