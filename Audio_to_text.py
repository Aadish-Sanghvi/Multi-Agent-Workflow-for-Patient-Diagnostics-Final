# ==========================================
# Install dependencies (for local testing)
# ==========================================
# pip install groq google-generativeai sounddevice soundfile

# ==========================================
# Imports
# ==========================================
import os
import tempfile
import numpy as np
import soundfile as sf
import wave
from groq import Groq
import google.generativeai as genai
import sounddevice as sd
from datetime import datetime
from config import gemini_api_key, groq_api_key
import threading
import signal

# ==========================================
# Groq Client Initialization
# ==========================================
try:
    client = Groq(api_key=groq_api_key)
    print("✅ Groq client initialized successfully!")
except Exception as e:
    raise Exception(f"Failed to initialize Groq client: {str(e)}")

# ==========================================
# Gemini Client Initialization (2.5 Flash)
# ==========================================
try:
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # Free tier model (adjust to gemini-2.5-flash if available)
    print("✅ Gemini 2.5 Flash API initialized successfully!")
except Exception as e:
    print(f"⚠️ Failed to initialize Gemini: {str(e)}. Fallback transcription disabled.")
    gemini_model = None

# ==========================================
# Functions
# ==========================================
def record_audio(fs=44100):
    """Record audio locally until stopped with Ctrl+C."""
    print("Press Enter to start recording...")
    input()  # Wait for Enter to start
    print("Recording... Press Ctrl+C to stop.")
    
    # Initialize audio buffer
    audio = []
    stop_event = threading.Event()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio.append(indata.copy())
    
    # Start recording stream
    stream = sd.InputStream(samplerate=fs, channels=1, callback=callback)
    stream.start()
    
    # Handle Ctrl+C to stop recording
    def signal_handler(sig, frame):
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while not stop_event.is_set():
            sd.sleep(100)  # Sleep to avoid busy-waiting
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
    
    print("Recording stopped.")
    
    # Save to temporary WAV file
    audio_data = np.concatenate(audio, axis=0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    
    return temp_file.name

def transcribe_groq_whisper(audio_file, language):
    """Transcribe using Groq's Whisper model for Hindi, English, Hinglish."""
    if audio_file is None:
        return "❌ No audio provided for Groq Whisper."

    try:
        lang_code = 'hi' if language.lower() in ['hindi', 'hinglish'] else 'en'
        with open(audio_file, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_file), f.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
                language=lang_code
            )
        return transcription
    except Exception as e:
        return f"❌ Groq Whisper error: {str(e)}"

def transcribe_gemini_fallback(audio_file, language):
    """Transcribe using Gemini 2.5 Flash for other languages or as fallback."""
    if audio_file is None or gemini_model is None:
        return "❌ No audio provided or Gemini model not available."

    try:
        # Convert audio to WAV if needed (Gemini supports WAV, MP3, M4A)
        data, samplerate = sf.read(audio_file)
        if len(data.shape) > 1:  # Stereo to Mono
            data = data.mean(axis=1)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            sf.write(temp_wav.name, data, samplerate)
            uploaded_file = genai.upload_file(temp_wav.name)
        
        prompt = f"Transcribe this audio in {language}. Provide only the transcription text."
        response = gemini_model.generate_content([uploaded_file, prompt])
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini 2.5 Flash error: {str(e)}"

def combined_transcription(audio_file=None, language="Hindi"):
    """Process audio based on selected language."""
    if audio_file is None:
        audio_file = record_audio()

    print(f"🔄 Processing audio file: {audio_file} in {language}")

    if language.lower() in ['hindi', 'english', 'hinglish']:
        groq_result = transcribe_groq_whisper(audio_file, language)
        gemini_result = "Not used (Groq Whisper handles Hindi/English/Hinglish)"
    else:
        groq_result = "Not used (Gemini handles other languages)"
        gemini_result = transcribe_gemini_fallback(audio_file, language)

    # Save results to file
    output_file = f"transcription_output_{datetime.utcnow().isoformat()}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Language: {language}\n")
        f.write(f"Groq Whisper Result: {groq_result}\n")
        f.write(f"Gemini 2.5 Flash Result: {gemini_result}\n")
    print(f"✅ Transcription saved to {output_file}")

    return groq_result, gemini_result

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Get user input for language
    valid_languages = ['Hindi', 'English', 'Hinglish', 'Spanish', 'French', 'Other']
    print("Available languages:", ", ".join(valid_languages))
    language = input("Enter the language for transcription: ").strip().capitalize()
    if language not in valid_languages:
        print(f"❌ Invalid language. Defaulting to Hindi.")
        language = "Hindi"

    # Get user input for recording or audio file
    use_recording = input("Record audio? (yes/no): ").strip().lower() == 'yes'
    if use_recording:
        audio_file = None
    else:
        audio_file = input("Enter path to audio file (e.g., audio.m4a): ").strip()
        if not os.path.exists(audio_file):
            print("❌ Audio file not found. Recording audio instead.")
            audio_file = None

    try:
        groq_result, gemini_result = combined_transcription(audio_file, language)
        print("\nTranscription Results:")
        print(f"Groq Whisper (Hindi/English/Hinglish): {groq_result}")
        print(f"Gemini 2.5 Flash (Other Languages): {gemini_result}")
    except Exception as e:
        print(f"❌ Error during transcription: {str(e)}")