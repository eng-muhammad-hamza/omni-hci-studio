"""OmniHCI Studio - UI Tabs Package"""
from .tab_chat import create_chat_tab
from .tab_vision import create_vision_tab
from .tab_mediapipe import create_mediapipe_tab
from .tab_audio import create_audio_tab
from .tab_nlp import create_nlp_tab
from .tab_pipeline import create_pipeline_tab

__all__ = [
    "create_chat_tab",
    "create_vision_tab",
    "create_mediapipe_tab",
    "create_audio_tab",
    "create_nlp_tab",
    "create_pipeline_tab",
]
