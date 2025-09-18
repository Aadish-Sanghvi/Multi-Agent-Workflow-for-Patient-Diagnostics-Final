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
                self.model_name, system_instruction=self.system_prompt
            )
            logger.info(f"Gemini model '{self.model_name}' initialized with system prompt!")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
            self.client = None

    def generate_content(self, prompt: str) -> Optional[str]:
        if not self.client:
            logger.error("Gemini model not available.")
            return None
        try:
            response = self.client.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini content generation error: {str(e)}")
            return None


class RoleSeparator:
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client

    def separate_roles(self, transcript: str, patient_id: str = "default") -> Dict[str, Any]:
        if not transcript or transcript.strip() == "":
            logger.error("No transcript provided.")
            return {"error": "No transcript provided."}

        prompt_text = role_separation_prompt_template.format(transcript=transcript)
        result_text = self.gemini_client.generate_content(prompt_text)
        if not result_text:
            return {"error": "Gemini model not available or failed to generate content."}

        result_json = self._parse_json(result_text)
        if not result_json:
            logger.error("Failed to parse Gemini response.")
            return {"error": "Failed to parse Gemini response."}

        # Single timestamp used everywhere
        timestamp = datetime.now(timezone.utc).isoformat()
        safe_timestamp = timestamp.replace(":", "_")

        if isinstance(result_json, list):
            output = {
                "conversation": result_json,
                "patient_id": patient_id,
                "timestamp": timestamp,
            }
        else:
            # If it's already a dict, just add metadata
            output = result_json
            output["patient_id"] = patient_id
            output["timestamp"] = timestamp

        self._save_output(output, patient_id, safe_timestamp)
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
    sample_transcript = (
        "hello doctor good morning actually na mujhe pichle ek hafte se khansi ho rahi hai aur kabhi kabhi bukhaar bhi aa jata hai hmm thoda detail mein bataiye khansi kaisi hai dry hai ya phlegm aa raha hai haan kabhi dry hai aur kabhi na thoda yellow type ka phlegm aata hai specially raat ko jab sota hu tab zyada problem hoti hai cough continuously chalta hai okay fever kitna high gaya hai aapne measure kiya kya haan kal raat ko thermometer se check kiya tha 101.2 aya tha aur body mein halka sa pain bhi hai headache bhi hai shortness of breath bhi feel ho raha hai matlab jab aap stairs chadte hain ya tez chal kar jaate hain yes doctor even jab main apne ghar ke second floor tak jata hoon to saans phool jata hai aur mujhe lagta hai ki main thoda weak ho gaya hu thik hai aap smoking karte hain ya pehle karte the ji main 10 saal tak cigarette pita tha but maine 2021 mein quit kar diya ab main bilkul nahi pita acha aur koi purana illness jaise asthma ya TB ya allergies kuch history hai nahi doctor aisa kuch nahi hai bas childhood mein ek baar pneumonia hua tha but uske baad kabhi problem nahi hui hmm samajh gaya abhi mujhe lag raha hai ki chest infection ho sakta hai pneumonia ya bronchitis aur aapki smoking history ki wajah se chronic bronchitis rule out karna padega doctor mujhe raat mein sote waqt bahut zyada cough hota hai is wajah se main so nahi pata hu kya turant kuch dawa le sakta hu haan abhi ke liye main aapko ek antibiotic likh raha hu aur ek cough syrup jo night mein lena hai saath mein paracetamol fever ke liye le lijiye warm water pijiyega aur steam inhalation bhi kijiye okay doctor x ray karwana zaroori hai kya haan main suggest karunga chest x ray aur ek spirometry lung function test karaiye jisse humein clear picture milegi ki infection hai ya COPD jaisa chronic issue theek hai doctor main kal hi test karwa lunga kya mujhe hospital admit hone ki zarurat hai abhi zarurat nahi hai lekin agar bukhaar 103 se upar jaye ya breathing aur zyada problem kare to turant emergency mein aaiye aur mujhe call kijiye"
    )

    gemini_client = GeminiClient(
        api_key=gemini_api_key,
        model_name=gemini_model_name,
        system_prompt=audio_text_to_chat_system_prompt,
    )
    role_separator = RoleSeparator(gemini_client)

    result = role_separator.separate_roles(sample_transcript, patient_id="PAT-101")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

