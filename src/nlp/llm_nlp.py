"""
OmniHCI Studio - LLM Zero-Shot NLP Engine
Leverages Ollama for zero-shot sentiment, intent detection, named entity extraction, and summarization.
"""
import json
import requests
from src.config import OLLAMA_MODEL, OLLAMA_URL

class OllamaNLP:
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_URL):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system: str = None) -> str:
        """Call Ollama via REST API for maximum reliability across environments."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            if system:
                payload["system"] = system

            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=45
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            return f"[Ollama Error: HTTP {resp.status_code}]"
        except Exception as e:
            return f"[Ollama Connection Error: {str(e)}]"

    def detect_sentiment(self, text: str) -> str:
        prompt = (
            f"Analyze the sentiment of this text. Reply with ONLY one word: "
            f"Positive, Negative, or Neutral.\n\nText: {text}"
        )
        res = self.generate(prompt)
        return res.strip().split("\n")[0].capitalize()

    def classify_intent(self, text: str) -> str:
        prompt = (
            f"Classify the primary intent of this statement into ONE category from: "
            f"Question, Complaint, Request, Greeting, Farewell, Feedback, Other.\n\n"
            f"Message: {text}\n\nReply with ONLY the category name."
        )
        res = self.generate(prompt)
        return res.strip().split("\n")[0]

    def extract_entities(self, text: str) -> str:
        prompt = (
            f"Extract named entities from this text. Identify People, Organizations, Locations, and Dates.\n"
            f"Provide a clean bulleted list.\n\nText: {text}"
        )
        return self.generate(prompt)

    def summarize(self, text: str, max_words: int = 50) -> str:
        prompt = f"Summarize the following passage in under {max_words} words:\n\n{text}"
        return self.generate(prompt)
