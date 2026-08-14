"""
OmniHCI Studio - TF-IDF & Naive Bayes Intent Classifier
Fast, robust statistical intent classification across core human-computer interaction commands and queries.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

CORE_TRAINING_CORPUS = [
    # Vision & Image Processing
    ("Can you analyze this image for me?", "vision_request"),
    ("Apply edge detection and filter out noise from this picture.", "vision_request"),
    ("What objects and colors do you observe in this frame?", "vision_request"),
    ("Process this photo using Gaussian blur and Otsu threshold.", "vision_request"),
    ("Extract the visual features and bit planes from this file.", "vision_request"),
    # Audio & Speech
    ("Transcribe this spoken recording into text.", "audio_request"),
    ("Can you remove the background noise and hum from this audio?", "audio_request"),
    ("Generate a Mel-spectrogram and compute the pitch and tempo.", "audio_request"),
    ("Synthesize this text into spoken voice audio.", "audio_request"),
    ("What is the signal to noise ratio of this recording?", "audio_request"),
    # Perception & Gestures
    ("Track the hand landmarks and detect what gesture is being shown.", "perception_request"),
    ("Is the person in the video looking tired or closing their eyes?", "perception_request"),
    ("Measure the eye aspect ratio and mouth opening for fatigue.", "perception_request"),
    ("Recognize the lip shape and vowel articulation.", "perception_request"),
    ("Detect facial emotion and eye blink patterns.", "perception_request"),
    # System Navigation & Help
    ("How do I use this application and its different tools?", "system_help"),
    ("What capabilities and features are available here?", "system_help"),
    ("Show me a tutorial or guide on how to process files.", "system_help"),
    ("Explain the difference between Butterworth filters and Fourier transforms.", "system_help"),
    ("How do I connect my local model or microphone?", "system_help"),
    # Greetings & Salutations
    ("Hello, how are you today?", "greeting"),
    ("Hi there, good morning!", "greeting"),
    ("Greetings assistant, are you ready to assist me?", "greeting"),
    ("Hey, nice to meet you.", "greeting"),
    # Feedback & Politeness
    ("Thank you very much, that was very helpful.", "appreciation"),
    ("Thanks for the quick and accurate answer!", "appreciation"),
    ("Goodbye, have a great day ahead.", "farewell"),
    ("See you later, closing the session now.", "farewell")
]

class IntentSentimentClassifier:
    def __init__(self, data=None):
        self.corpus = data if data else CORE_TRAINING_CORPUS
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("clf", MultinomialNB(alpha=0.4))
        ])
        self.is_trained = False
        self.accuracy = 0.0
        self.report_text = ""
        self._train()

    def _train(self):
        texts = [item[0] for item in self.corpus]
        labels = [item[1] for item in self.corpus]
        
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.25, random_state=42, stratify=None
        )
        
        self.pipeline.fit(X_train, y_train)
        preds = self.pipeline.predict(X_test)
        self.accuracy = round(accuracy_score(y_test, preds) * 100, 1)
        self.report_text = classification_report(y_test, preds, zero_division=0)
        
        # Fit on full dataset for comprehensive deployment coverage
        self.pipeline.fit(texts, labels)
        self.is_trained = True

    def predict(self, text: str) -> tuple:
        """Classify input text. Returns (intent_label, confidence_percentage)."""
        if not text or not text.strip():
            return "unknown", 0.0
        label = self.pipeline.predict([text])[0]
        probs = self.pipeline.predict_proba([text])[0]
        classes = list(self.pipeline.classes_)
        conf = float(probs[classes.index(label)])
        return label, round(conf * 100, 1)

    def get_metrics(self) -> dict:
        return {
            "validation_accuracy": f"{self.accuracy}%",
            "classes": list(self.pipeline.classes_),
            "classification_report": self.report_text
        }
