"""
OmniHCI Studio - Tab 3: MediaPipe Perception & Affective Computing Suite
"""
import gradio as gr
from src.mediapipe.hands import process_hand_gesture
from src.mediapipe.face_mesh import process_face_mesh
from src.mediapipe.lips import process_lip_vowel

def create_mediapipe_tab():
    with gr.Tab("👁️ MediaPipe & Affective Perception"):
        gr.Markdown(
            "### 🧬 Perceptual Computing & Human-Computer Interface Suite\n"
            "Real-time geometric landmark analysis for Hand Gestures, Facial Affect / Drowsiness, and Lip Vowel Articulation."
        )

        with gr.Tabs():
            # Sub-Tab 1: Hand Recognition
            with gr.Tab("🤚 Hand Gestures & Finger Tracking"):
                with gr.Row():
                    with gr.Column(scale=1):
                        hand_img = gr.Image(sources=["upload", "webcam"], type="pil", label="Hand Image")
                        hand_btn = gr.Button("🖐️ Detect Gestures", variant="primary")
                    with gr.Column(scale=1):
                        hand_out = gr.Image(type="numpy", label="Landmark Visualization")
                        hand_info = gr.JSON(label="Hand Geometry Metrics")
                        hand_stat = gr.Markdown("Ready to track hand gestures.")

                hand_btn.click(
                    process_hand_gesture,
                    inputs=[hand_img],
                    outputs=[hand_out, hand_info, hand_stat]
                )

            # Sub-Tab 2: Face Mesh / EAR / MAR
            with gr.Tab("😊 Face Mesh, Drowsiness & Yawn Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        face_img = gr.Image(sources=["upload", "webcam"], type="pil", label="Face Image")
                        face_btn = gr.Button("😊 Analyze Face & Fatigue", variant="primary")
                    with gr.Column(scale=1):
                        face_out = gr.Image(type="numpy", label="Mesh Tessellation & EAR/MAR")
                        face_info = gr.JSON(label="Facial Metric Breakdown")
                        face_stat = gr.Markdown("Ready for face inspection.")

                face_btn.click(
                    process_face_mesh,
                    inputs=[face_img],
                    outputs=[face_out, face_info, face_stat]
                )

            # Sub-Tab 3: Lip Reading & Vowels
            with gr.Tab("👄 Lip Tracking & Vowel Articulation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        lip_img = gr.Image(sources=["upload", "webcam"], type="pil", label="Face / Lip Image")
                        lip_btn = gr.Button("👄 Track Lips & Classify Vowel", variant="primary")
                    with gr.Column(scale=1):
                        lip_out = gr.Image(type="numpy", label="Lip Contours")
                        lip_info = gr.JSON(label="Aperture & Geometry Metrics")
                        lip_stat = gr.Markdown("Ready for lip tracking.")

                lip_btn.click(
                    process_lip_vowel,
                    inputs=[lip_img],
                    outputs=[lip_out, lip_info, lip_stat]
                )
