"""
OmniHCI Studio - Automated Speech Recognition (ASR) Engine
Supports Whisper, format conversion (MP3 -> WAV), and acoustic signal conditioning.
"""
import os
import tempfile
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from src.config import WHISPER_SIZE, SAMPLE_RATE

_whisper_cache = None

def get_whisper_model(model_name: str = WHISPER_SIZE):
    global _whisper_cache
    if _whisper_cache is None:
        try:
            import whisper
            _whisper_cache = whisper.load_model(model_name)
        except Exception as e:
            print(f"[ASR] Error loading Whisper model '{model_name}': {e}")
            return None
    return _whisper_cache

def ensure_wav(audio_path: str, target_sr: int = SAMPLE_RATE) -> str:
    """Ensure audio file is in standard 16kHz mono WAV format."""
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    ext = os.path.splitext(audio_path)[1].lower()
    if ext == ".wav":
        return audio_path

    # Convert via PyDub
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav.close()
    
    try:
        seg = AudioSegment.from_file(audio_path)
        seg = seg.set_frame_rate(target_sr).set_channels(1)
        seg.export(temp_wav.name, format="wav")
        return temp_wav.name
    except Exception as e:
        print(f"[ASR] Conversion error for {audio_path}: {e}")
        return audio_path

def load_audio_normalized(audio_path: str, target_sr: int = SAMPLE_RATE) -> tuple:
    """Load WAV audio as mono float32 numpy array and sample rate."""
    wav_path = ensure_wav(audio_path, target_sr)
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Condition signal: DC offset removal + pre-emphasis + normalization
    audio -= np.mean(audio)
    if len(audio) > 1:
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio, sr

def transcribe_audio(audio_path: str, model_size: str = WHISPER_SIZE, language: str = None) -> str:
    """Transcribe audio file using Whisper."""
    if not audio_path or not os.path.exists(audio_path):
        return ""

    try:
        wav_path = ensure_wav(audio_path)
        model = get_whisper_model(model_size)
        if model is None:
            return "[Error: Whisper model unavailable. Ensure openai-whisper is installed.]"

        kwargs = {"fp16": False}
        if language:
            kwargs["language"] = language

        result = model.transcribe(wav_path, **kwargs)
        return result.get("text", "").strip()
    except Exception as e:
        return f"[Transcription Error: {str(e)}]"
