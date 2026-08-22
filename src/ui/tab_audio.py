"""
OmniHCI Studio - Tab 4: Audio Engineering & Acoustics Lab
"""
import gradio as gr
from src.audio.dsp import process_audio_dsp, extract_librosa_features, generate_spectrogram_plot
from src.audio.tts import synthesize_speech, list_system_voices

def create_audio_tab():
    with gr.Tab("🎵 Audio Engineering & Acoustics"):
        gr.Markdown(
            "### 🎚️ Speech Signal Processing, Acoustic Features & TTS\n"
            "Filter audio, inspect Mel-spectrograms & MFCCs, estimate SNR, and synthesize speech offline."
        )

        with gr.Tabs():
            # Sub-Tab 1: DSP & Filtering
            with gr.Tab("🎛️ Signal Filtering & DSP"):
                with gr.Row():
                    with gr.Column(scale=1):
                        aud_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Audio")
                        filter_type = gr.Radio(
                            choices=["bandpass", "lowpass", "highpass"],
                            value="bandpass",
                            label="Butterworth Filter Type"
                        )
                        vad_check = gr.Checkbox(label="Enable Voice Activity Detection (VAD)", value=False)
                        dsp_btn = gr.Button("⚡ Apply DSP Pipeline", variant="primary")

                    with gr.Column(scale=1):
                        aud_out = gr.Audio(label="Processed Audio (Cleaned)", type="filepath")
                        snr_info = gr.Markdown("Audio metrics will appear here.")

                def run_dsp(path, f_type, vad_on):
                    if not path:
                        return None, "⚠️ Please provide an audio input file or recording."
                    try:
                        clean_path, snr_in, snr_out, dur = process_audio_dsp(path, filter_type=f_type, do_vad=vad_on)
                        report = (
                            f"**Initial SNR:** `{snr_in} dB`\n\n"
                            f"**Enhanced SNR:** `{snr_out} dB`\n\n"
                            f"**Duration:** `{dur}s`\n\n"
                            f"**Filter applied:** `{f_type.upper()}` (Order 5)"
                        )
                        return clean_path, report
                    except Exception as e:
                        return None, f"❌ DSP error: {e}"

                dsp_btn.click(run_dsp, inputs=[aud_in, filter_type, vad_check], outputs=[aud_out, snr_info])

            # Sub-Tab 2: Spectral & Librosa Features
            with gr.Tab("📊 Spectral Analysis & MFCCs"):
                with gr.Row():
                    with gr.Column(scale=1):
                        spec_audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio Input")
                        spec_btn = gr.Button("📈 Extract Features & Spectrogram", variant="primary")
                        feat_json = gr.JSON(label="Extracted Acoustic Features")
                    with gr.Column(scale=1):
                        spec_img = gr.Image(type="filepath", label="Waveform & Mel-Spectrogram")

                def run_spectral(path):
                    if not path:
                        return None, {"error": "No audio file provided"}
                    try:
                        img_path = generate_spectrogram_plot(path)
                        features = extract_librosa_features(path)
                        return img_path, features
                    except Exception as e:
                        return None, {"error": str(e)}

                spec_btn.click(run_spectral, inputs=[spec_audio_in], outputs=[spec_img, feat_json])

            # Sub-Tab 3: Text to Speech (TTS)
            with gr.Tab("🗣️ Offline Text-to-Speech Studio"):
                voices = list_system_voices()
                voice_choices = [f"Voice {i}: {v['name']}" for i, v in enumerate(voices)] or ["Default System Voice"]

                with gr.Row():
                    with gr.Column(scale=1):
                        tts_text = gr.Textbox(
                            label="Text to Synthesize",
                            placeholder="Type any sentence here to convert into spoken audio...",
                            lines=3,
                            value="Welcome to OmniHCI Studio, the unified multimodal human-computer interaction platform."
                        )
                        voice_drop = gr.Dropdown(choices=voice_choices, value=voice_choices[0], label="Select Voice")
                        rate_slider = gr.Slider(minimum=100, maximum=250, value=165, step=5, label="Speaking Rate (WPM)")
                        vol_slider = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, label="Volume")
                        speak_btn = gr.Button("🔊 Synthesize Speech", variant="primary")

                    with gr.Column(scale=1):
                        tts_audio_out = gr.Audio(label="Synthesized WAV Audio", type="filepath")

                def run_tts(txt, v_choice, rate, vol):
                    if not txt:
                        return None
                    v_idx = 0
                    if "Voice " in v_choice:
                        try:
                            v_idx = int(v_choice.split(":")[0].replace("Voice ", ""))
                        except Exception:
                            v_idx = 0
                    return synthesize_speech(txt, rate=int(rate), volume=float(vol), voice_index=v_idx)

                speak_btn.click(run_tts, inputs=[tts_text, voice_drop, rate_slider, vol_slider], outputs=[tts_audio_out])
