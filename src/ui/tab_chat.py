"""
OmniHCI Studio - Tab 1: Multimodal Conversational Agent
"""
import gradio as gr
from src.agent.multimodal_agent import MultimodalAgent

def create_chat_tab(agent: MultimodalAgent):
    with gr.Tab("💬 Conversational AI Agent"):
        gr.Markdown(
            "### 💬 Multimodal Conversational Assistant\n"
            "Interact naturally through **Text**, **Voice** (microphone / audio upload), or **Vision** (camera capture / image upload). Supports unified multi-input reasoning."
        )

        with gr.Row():
            # Left Column: Inputs
            with gr.Column(scale=1):
                txt_input = gr.Textbox(
                    label="📝 Text Query",
                    placeholder="Type a message or instruction here...",
                    lines=3
                )
                with gr.Row():
                    char_count = gr.Markdown("Characters: **0** / 500")

                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="🎙️ Voice Input"
                )

                image_input = gr.Image(
                    sources=["upload", "webcam"],
                    type="pil",
                    label="📷 Visual Frame / Document"
                )

                tts_toggle = gr.Checkbox(
                    label="🔊 Vocalize response aloud (Offline TTS)",
                    value=False
                )

                with gr.Row():
                    send_btn = gr.Button("🚀 Submit Query", variant="primary")
                    clear_btn = gr.Button("🗑️ Reset")

            # Right Column: Chatbot & Output
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Dialogue Stream", height=420)
                status_box = gr.Textbox(
                    label="Modality & Intent Status",
                    interactive=False,
                    value="Ready for input."
                )
                audio_output = gr.Audio(
                    label="🔊 Spoken Output",
                    type="filepath",
                    interactive=False
                )

        # State management
        history_state = gr.State([])
        is_processing = gr.State(False)

        def update_chars(t):
            cnt = len(t or "")
            return f"Characters: **{cnt}** / 500"

        txt_input.change(update_chars, inputs=[txt_input], outputs=[char_count])

        def on_send(txt, aud, img, hist, tts_on, busy):
            if busy:
                return hist, hist, "Processing previous query, please wait...", None, busy
            busy = True
            try:
                new_hist, badge, aud_out, stat = agent.chat_step(
                    text_input=txt,
                    audio_path=aud,
                    image_input=img,
                    history=hist,
                    enable_tts=tts_on
                )
                return new_hist, new_hist, stat, aud_out, False
            except Exception as e:
                return hist, hist, f"Error: {e}", None, False

        send_btn.click(
            on_send,
            inputs=[txt_input, audio_input, image_input, history_state, tts_toggle, is_processing],
            outputs=[history_state, chatbot, status_box, audio_output, is_processing]
        )

        def on_clear():
            return [], [], "", None, None, "Session reset."

        clear_btn.click(
            on_clear,
            inputs=[],
            outputs=[history_state, chatbot, txt_input, audio_input, image_input, status_box]
        )
