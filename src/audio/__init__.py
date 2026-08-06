"""OmniHCI Studio - Audio Engineering & Speech Processing Module"""
from .asr import transcribe_audio, load_audio_normalized
from .dsp import (
    process_audio_dsp,
    extract_librosa_features,
    generate_spectrogram_plot,
)
from .tts import synthesize_speech, list_system_voices

__all__ = [
    "transcribe_audio",
    "load_audio_normalized",
    "process_audio_dsp",
    "extract_librosa_features",
    "generate_spectrogram_plot",
    "synthesize_speech",
    "list_system_voices",
]
