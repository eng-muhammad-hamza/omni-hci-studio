"""
OmniHCI Studio - Tab 6: End-to-End Full Loop Pipeline
Mic/Audio -> Whisper ASR -> NLTK/ML Intent -> Ollama LLM Reasoning -> pyttsx3 TTS
"""
import gradio as gr
from src.audio.asr import transcribe_audio
from src.nlp.nltk_tools import run_nltk_pipeline
from src.nlp.classifier import IntentSentimentClassifier
from src.nlp.llm_nlp import OllamaNLP
from src.audio.tts import synthesize_speech

def create_pipeline_tab():
    classifier = IntentSentimentClassifier()
    llm = OllamaNLP()

    with gr.Tab("🔗 End-to-End Multimodal Pipeline"):
        gr.Markdown(
            "### 🔄 The Continuous Multimodal Loop\n"
            "Chains **Acoustic Input (Mic)** → **ASR (Whisper)** → **NLP Intent & Syntax (NLTK/ML)** → **Cognitive Reasoning (Ollama)** → **Speech Synthesis (TTS)**."
        )

        with gr.Row():
            with gr.Column(scale=1):
                mic_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎙️ Step 1: Voice Input")
                run_btn = gr.Button("⚡ Execute Full HCI Loop", variant="primary")

            with gr.Column(scale=2):
                with gr.Accordion("Stage 1 — Speech Recognition (Whisper)", open=True):
                    stage1_txt = gr.Textbox(label="Transcribed Spoken Text", interactive=False)

                with gr.Accordion("Stage 2 — Linguistic & Intent Analysis (NLTK + ML)", open=True):
                    stage2_txt = gr.Markdown("Intent & syntactic breakdown will appear here.")

                with gr.Accordion("Stage 3 — Cognitive Response (Ollama)", open=True):
                    stage3_txt = gr.Textbox(label="AI Reasoning Response", lines=3, interactive=False)

                with gr.Accordion("Stage 4 — Speech Output (pyttsx3)", open=True):
                    stage4_audio = gr.Audio(label="Synthesized Vocal Reply", type="filepath", interactive=False)

        def run_full_loop(audio_path):
            if not audio_path:
                return "No audio provided.", "N/A", "Please record or upload audio.", None

            # 1. ASR
            transcript = transcribe_audio(audio_path)
            if not transcript:
                transcript = "[Could not transcribe audio or audio was silent]"

            # 2. NLP Analysis
            intent, conf = classifier.predict(transcript)
            nltk_res = run_nltk_pipeline(transcript)
            stage2_report = (
                f"**Predicted Intent:** `{intent.upper()}` ({conf}% confidence)\n\n"
                f"**Keywords:** {', '.join(nltk_res['clean_tokens'][:6])}\n\n"
                f"**Entities:** {', '.join(nltk_res['named_entities']) or 'None'}"
            )

            # 3. LLM Reasoning
            prompt = (
                f"You are a helpful, empathetic assistant responding to a user who spoke to you.\n"
                f"User statement: {transcript}\n"
                f"Detected Intent: {intent}\n"
                f"Provide a clear, helpful 2-sentence response:"
            )
            llm_reply = llm.generate(prompt)

            # 4. TTS
            tts_audio = synthesize_speech(llm_reply)

            return transcript, stage2_report, llm_reply, tts_audio

        run_btn.click(
            run_full_loop,
            inputs=[mic_input],
            outputs=[stage1_txt, stage2_txt, stage3_txt, stage4_audio]
        )
