"""
OmniHCI Studio - Offline Text-to-Speech (TTS) Engine
Uses pyttsx3 for cross-platform, latency-free vocal feedback.
"""
import os
import tempfile

def list_system_voices() -> list:
    """Retrieve all available system TTS voices."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        voice_list = [{"id": v.id, "name": v.name} for v in voices]
        engine.stop()
        return voice_list
    except Exception as e:
        print(f"[TTS] Voice listing error: {e}")
        return []

def synthesize_speech(text: str, rate: int = 165, volume: float = 1.0, voice_index: int = 0) -> str:
    """
    Synthesize text into a WAV file and return file path.
    """
    if not text or not text.strip():
        return None

    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        
        engine.setProperty("rate", rate)
        engine.setProperty("volume", max(0.0, min(1.0, volume)))
        if voices and 0 <= voice_index < len(voices):
            engine.setProperty("voice", voices[voice_index].id)

        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()

        engine.save_to_file(text, tmp_wav.name)
        engine.runAndWait()
        engine.stop()
        return tmp_wav.name
    except Exception as e:
        print(f"[TTS] Synthesis failed: {e}")
        return None
