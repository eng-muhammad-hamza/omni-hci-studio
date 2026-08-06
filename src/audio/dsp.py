"""
OmniHCI Studio - Audio DSP & Acoustics Engine
Signal filtering, Voice Activity Detection, Librosa feature extraction, and spectral visualization.
"""
import os
import tempfile
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    return audio - np.mean(audio)

def apply_preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    if len(audio) < 2:
        return audio
    return np.append(audio[0], audio[1:] - coeff * audio[:-1])

def bandpass_filter(audio: np.ndarray, sr: int, low_hz: float = 300.0, high_hz: float = 3400.0, order: int = 5) -> np.ndarray:
    nyq = sr / 2.0
    low = max(0.01, min(0.99, low_hz / nyq))
    high = max(0.01, min(0.99, high_hz / nyq))
    if low >= high:
        return audio
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, audio)

def lowpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 4000.0, order: int = 5) -> np.ndarray:
    nyq = sr / 2.0
    cut = max(0.01, min(0.99, cutoff_hz / nyq))
    sos = butter(order, cut, btype="low", output="sos")
    return sosfilt(sos, audio)

def highpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 80.0, order: int = 5) -> np.ndarray:
    nyq = sr / 2.0
    cut = max(0.01, min(0.99, cutoff_hz / nyq))
    sos = butter(order, cut, btype="high", output="sos")
    return sosfilt(sos, audio)

def estimate_snr(audio: np.ndarray, noise_samples: int = 1600) -> float:
    if len(audio) <= noise_samples:
        noise_samples = max(10, len(audio) // 4)
    noise = audio[:noise_samples]
    sig_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-12:
        return 50.0
    return float(10.0 * np.log10(sig_power / noise_power))

def apply_vad(audio: np.ndarray, sr: int, threshold_factor: float = 0.25) -> np.ndarray:
    """Extract voiced segments using RMS energy VAD."""
    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.010)
    frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len)
    energy = np.sqrt(np.mean(frames ** 2, axis=0))
    threshold = threshold_factor * np.max(energy)
    voiced = []
    for i, e in enumerate(energy):
        if e > threshold:
            start = i * hop_len
            end = start + frame_len
            voiced.append(audio[start:end])
    return np.concatenate(voiced) if voiced else audio

def process_audio_dsp(audio_path: str, filter_type: str = "bandpass", do_vad: bool = False) -> tuple:
    """
    Applies complete DSP chain and saves processed WAV to temp file.
    Returns: (output_wav_path, snr_before, snr_after, duration)
    """
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    initial_snr = estimate_snr(audio)
    
    # 1. DC Offset
    proc = remove_dc_offset(audio)
    # 2. Filtering
    if filter_type == "bandpass":
        proc = bandpass_filter(proc, sr, 300.0, 3400.0)
    elif filter_type == "lowpass":
        proc = lowpass_filter(proc, sr, 4000.0)
    elif filter_type == "highpass":
        proc = highpass_filter(proc, sr, 100.0)
    # 3. Pre-emphasis
    proc = apply_preemphasis(proc)
    # 4. Optional VAD
    if do_vad:
        proc = apply_vad(proc, sr)
    # 5. Peak normalize
    peak = np.max(np.abs(proc))
    if peak > 0:
        proc /= peak

    final_snr = estimate_snr(proc)
    duration = len(proc) / sr

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, proc, sr)
    return tmp.name, round(initial_snr, 2), round(final_snr, 2), round(duration, 2)

def extract_librosa_features(audio_path: str) -> dict:
    """Extract comprehensive acoustic and musical features."""
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Spectral Centroid & Rolloff
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)
    
    # Tempo / BPM
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo.item()) if tempo.size == 1 else float(tempo[0])

    # 13 MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    return {
        "duration_sec": round(len(audio) / sr, 2),
        "sample_rate": sr,
        "tempo_bpm": round(float(tempo), 1),
        "mean_spectral_centroid_hz": round(float(np.mean(centroid)), 1),
        "mean_spectral_rolloff_hz": round(float(np.mean(rolloff)), 1),
        "mean_zcr": round(float(np.mean(zcr)), 4),
        "mean_rms_energy": round(float(np.mean(rms)), 4),
        "mfcc_mean_vector": [round(float(v), 2) for v in np.mean(mfcc, axis=1)]
    }

def generate_spectrogram_plot(audio_path: str) -> str:
    """Generate a combined Waveform + Mel-Spectrogram image."""
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    plt.subplots_adjust(hspace=0.35)

    # Waveform
    time_axis = np.linspace(0, len(audio) / sr, len(audio))
    ax1.plot(time_axis, audio, color="#2563eb", linewidth=0.7)
    ax1.set_title("Waveform (Amplitude vs Time)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    # Mel Spectrogram
    melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    img = ax2.imshow(melspec_db, origin="lower", aspect="auto", cmap="magma",
                     extent=[0, len(audio) / sr, 0, sr / 2])
    ax2.set_title("Mel-Spectrogram (Frequency vs Time)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax2, format="%+2.0f dB")

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return tmp.name
