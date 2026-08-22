"""
OmniHCI Studio - Tab 2: Computer Vision Lab
"""
import gradio as gr
from src.vision.filters import apply_all_filters

FILTER_OPTIONS = [
    "Grayscale",
    "Negative (Invert)",
    "Binary Threshold",
    "Otsu Automatic Threshold",
    "Adaptive Gaussian Threshold",
    "Linear Contrast Stretch",
    "Gamma Correction",
    "Log Transform",
    "Z-Score Normalization",
    "Gaussian Blur",
    "Median Blur",
    "Average Blur",
    "Canny Edge Detection",
    "Sobel Edge Detection",
    "Laplacian Edge Detection",
    "Morphological Dilation",
    "Morphological Erosion",
    "Morphological Opening",
    "Morphological Closing",
    "Morphological Gradient",
    "Bit Plane MSB (Bit 7)",
    "Bit Plane LSB (Bit 0)",
    "Reconstruct from 4 MSBs",
]

def create_vision_tab():
    with gr.Tab("🖼️ Computer Vision Lab"):
        gr.Markdown(
            "### 🔬 Computer Vision & Digital Image Processing\n"
            "Apply classical image transformations, morphology, spatial filtering, and bit-plane analysis."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(sources=["upload", "webcam"], type="pil", label="Original Image")
                filter_choice = gr.Dropdown(choices=FILTER_OPTIONS, value="Grayscale", label="Select Operation")
                
                with gr.Accordion("Fine-Tuning Parameters", open=False):
                    ksize_slider = gr.Slider(minimum=3, maximum=21, step=2, value=5, label="Kernel / Block Size")
                    gamma_slider = gr.Slider(minimum=0.1, maximum=3.0, step=0.1, value=0.5, label="Gamma Exponent")
                    thresh_slider = gr.Slider(minimum=0, maximum=255, step=1, value=127, label="Binary Threshold Level")

                apply_btn = gr.Button("⚡ Process Image", variant="primary")

            with gr.Column(scale=1):
                output_img = gr.Image(type="numpy", label="Processed Output Image")
                status_txt = gr.Markdown("Select an image and filter to view results.")

        def run_filter(img, op, ksize, gamma, thresh):
            if img is None:
                return None, "⚠️ Please upload or capture an image first."
            try:
                res = apply_all_filters(
                    img, op,
                    ksize=int(ksize),
                    gamma=float(gamma),
                    thresh=int(thresh)
                )
                return res, f"✅ Successfully applied **{op}**."
            except Exception as e:
                return None, f"❌ Error processing image: {e}"

        apply_btn.click(
            run_filter,
            inputs=[input_img, filter_choice, ksize_slider, gamma_slider, thresh_slider],
            outputs=[output_img, status_txt]
        )
