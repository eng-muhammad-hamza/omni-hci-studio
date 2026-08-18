"""
OmniHCI Studio - Multimodal Conversational Agent
Unifies Text, Voice (Whisper), and Computer Vision (OpenCV/PIL) into a cohesive dialogue interface.
"""
import base64
import cv2
import numpy as np
import requests
from PIL import Image
from src.config import OLLAMA_MODEL, OLLAMA_URL, MAX_HISTORY
from src.audio.asr import transcribe_audio
from src.audio.tts import synthesize_speech

INTENT_DOMAINS = {
    "vision_analysis": [
        "image", "picture", "photo", "see", "look", "detect", "color", "visual", "inspect", "frame"
    ],
    "audio_speech": [
        "audio", "voice", "sound", "listen", "hear", "transcribe", "noise", "speech", "waveform", "mic"
    ],
    "system_assistance": [
        "help", "how", "what", "guide", "explain", "features", "capabilities", "support", "tutorial"
    ],
    "nlp_language": [
        "text", "grammar", "sentiment", "summary", "translate", "words", "meaning", "language"
    ],
    "general_inquiry": [
        "who", "when", "where", "why", "calculate", "tell", "describe"
    ]
}

SYSTEM_PROMPT = """You are the OmniHCI Studio Intelligent Assistant, an advanced multimodal AI designed to interact across text, speech, and computer vision.

Capabilities:
1. Visual Understanding: When an image or camera frame is provided, describe key visual objects, scene attributes, text, or quality.
2. Spoken Dialogue: Understand audio transcripts and communicate naturally.
3. System Guidance: Help users navigate Computer Vision tools, MediaPipe perception, Audio DSP, and NLP engines.

Guidelines:
- Provide clear, direct, and insightful responses.
- If only an image is uploaded without text, provide a concise visual summary.
- Maintain an encouraging, professional, and knowledgeable persona.
"""

class MultimodalAgent:
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_URL, max_history: int = MAX_HISTORY):
        self.model = model
        self.base_url = base_url
        self.max_history = max_history

    def classify_intent(self, text: str) -> str:
        """Classify user query into functional intent domains."""
        t_lower = text.lower()
        matched = []
        for domain, keywords in INTENT_DOMAINS.items():
            if any(kw in t_lower for kw in keywords):
                matched.append(domain)
        
        if not matched:
            if any(w in t_lower for w in ["hi", "hello", "hey", "greetings", "good morning", "good evening"]):
                return "greeting"
            return "general_query"
        return ", ".join(matched)

    def encode_image_b64(self, img_input) -> str:
        """Encode image to base64 JPEG."""
        if img_input is None:
            return None
        if isinstance(img_input, Image.Image):
            arr = np.array(img_input)
        else:
            arr = img_input

        if arr.ndim == 3 and arr.shape[2] == 3:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            bgr = arr

        bgr_resized = cv2.resize(bgr, (640, 480))
        _, buffer = cv2.imencode(".jpg", bgr_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")

    def trim_history(self, history: list) -> list:
        if len(history) > self.max_history:
            return history[-self.max_history:]
        return history

    def chat_step(self, text_input: str, audio_path: str, image_input, history: list, enable_tts: bool = False) -> tuple:
        """
        Processes a multimodal conversational turn.
        Returns: (updated_history, modality_badge, audio_response_path, status_msg)
        """
        history = list(history or [])
        modalities_used = []

        # 1. Voice transcription
        audio_transcript = ""
        if audio_path:
            audio_transcript = transcribe_audio(audio_path)
            if audio_transcript:
                modalities_used.append("VOICE")

        # 2. Text input
        clean_text = (text_input or "").strip()
        if clean_text:
            modalities_used.append("TEXT")

        # 3. Vision input
        image_b64 = None
        if image_input is not None:
            image_b64 = self.encode_image_b64(image_input)
            if image_b64:
                modalities_used.append("VISION")

        # Validation: Allow any modality to proceed
        if not modalities_used:
            return history, "NONE", None, "Please provide at least one input: type text, record voice, or provide an image."

        parts = []
        if clean_text:
            parts.append(clean_text)
        if audio_transcript:
            parts.append(f"[Spoken Input]: {audio_transcript}")
        combined_query = "\n".join(parts) if parts else "[User submitted an image without text]"

        # Modality badge
        if len(modalities_used) > 1:
            modality_badge = f"COMBINED ({' + '.join(modalities_used)})"
        else:
            modality_badge = f"{modalities_used[0]}"

        # Intent detection
        intent_label = self.classify_intent(combined_query)

        # Assemble Ollama Chat Payload
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user_msg, bot_msg in self.trim_history(history):
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})

        curr_msg = {"role": "user", "content": combined_query}
        if image_b64:
            curr_msg["images"] = [image_b64]
        messages.append(curr_msg)

        # Query local LLM
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                },
                timeout=60
            )
            if resp.status_code == 200:
                answer = resp.json().get("message", {}).get("content", "").strip()
            else:
                answer = f"Ollama returned HTTP error {resp.status_code}. Please ensure model '{self.model}' is available."
        except Exception as e:
            answer = f"Could not connect to Ollama ({str(e)}). Ensure `ollama serve` is active."

        history.append((combined_query, answer))

        # Optional audio reply via pyttsx3
        audio_out = None
        if enable_tts and answer:
            short_answer = ". ".join(answer.split(". ")[:3])
            audio_out = synthesize_speech(short_answer)

        status_msg = f"Processed | Modality: {modality_badge} | Focus: {intent_label}"
        return history, modality_badge, audio_out, status_msg
