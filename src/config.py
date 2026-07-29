"""
OmniHCI Studio - Central Configuration and Runtime Environment
"""
import os
import cv2
import requests
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

# Model & Engine Configurations
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "base")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))

def check_ollama_status() -> dict:
    """Check if the local Ollama daemon is reachable and list models."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if resp.status_code == 200:
            models = [m.get("name") for m in resp.json().get("models", [])]
            return {"online": True, "models": models, "error": None}
    except Exception as e:
        return {"online": False, "models": [], "error": str(e)}
    return {"online": False, "models": [], "error": "Unknown status"}

def check_camera_availability(camera_index: int = 0) -> bool:
    """Soft check for camera availability without throwing hard errors."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return False
    ret, frame = cap.read()
    cap.release()
    return bool(ret and frame is not None)

def get_system_health() -> dict:
    """Compile diagnostic status of external runtimes and hardware."""
    ollama_info = check_ollama_status()
    camera_ok = check_camera_availability(0)
    return {
        "ollama_online": ollama_info["online"],
        "ollama_models": ollama_info["models"],
        "camera_available": camera_ok,
        "default_model": OLLAMA_MODEL,
        "whisper_size": WHISPER_SIZE,
    }
