"""
OmniHCI Studio - Main Application Launcher
Unified Multimodal Human-Computer Interaction Platform
"""
import os
import sys
import gradio as gr
from src.config import get_system_health, SERVER_PORT
from src.agent.multimodal_agent import MultimodalAgent
from src.ui.tab_chat import create_chat_tab
from src.ui.tab_vision import create_vision_tab
from src.ui.tab_mediapipe import create_mediapipe_tab
from src.ui.tab_audio import create_audio_tab
from src.ui.tab_nlp import create_nlp_tab
from src.ui.tab_pipeline import create_pipeline_tab

def build_app():
    # Diagnostics check
    health = get_system_health()
    ollama_badge = "🟢 Ollama Connected" if health["ollama_online"] else "🔴 Ollama Offline (Run `ollama serve`)"
    camera_badge = "🟢 Camera Detected" if health["camera_available"] else "🟡 Camera Unavailable (Soft Warning)"

    custom_css = """
    .gradio-container { max-width: 1400px !important; margin: auto; }
    .status-badge { display: inline-block; padding: 4px 12px; border-radius: 9999px; font-weight: bold; font-size: 0.85rem; }
    """

    agent = MultimodalAgent()

    with gr.Blocks(title="OmniHCI Studio", theme=gr.themes.Soft(), css=custom_css) as demo:
        # Header & Diagnostics Bar
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(
                    "# 🌌 OmniHCI Studio\n"
                    "### Unified Platform for Multimodal Interaction, Affective Perception, Signal DSP & Natural Language"
                )
            with gr.Column(scale=1):
                gr.Markdown(
                    f"**System Telemetry:**\n\n"
                    f"- {ollama_badge}\n"
                    f"- {camera_badge}\n"
                    f"- 🤖 Model: `{health['default_model']}` | 🎙️ ASR: `{health['whisper_size']}`"
                )

        # 6 Core Functional Tabs
        with gr.Tabs():
            create_chat_tab(agent)
            create_vision_tab()
            create_mediapipe_tab()
            create_audio_tab()
            create_nlp_tab()
            create_pipeline_tab()

        gr.Markdown(
            "---\n"
            "<div style='text-align: center; color: gray; font-size: 0.85rem;'>"
            "OmniHCI Studio • Human-Computer Interaction Laboratory • Powered by Gradio, MediaPipe, OpenCV, Librosa, Whisper, NLTK & Ollama"
            "</div>"
        )

    return demo

if __name__ == "__main__":
    print("=" * 60)
    print("  Starting OmniHCI Studio...")
    print("=" * 60)
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=SERVER_PORT, show_error=True)
