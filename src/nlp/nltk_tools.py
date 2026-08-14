"""
OmniHCI Studio - Classical NLTK Text Processing Pipeline
"""
import string
import re
from collections import Counter
import nltk

# Ensure resources
NLTK_RESOURCES = [
    "punkt", "punkt_tab", "stopwords", "wordnet",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker", "maxent_ne_chunker_tab",
    "words", "omw-1.4"
]

for res in NLTK_RESOURCES:
    try:
        nltk.download(res, quiet=True)
    except Exception:
        pass

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
try:
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()

def run_nltk_pipeline(text: str) -> dict:
    """Run full analytical NLTK pipeline on input text."""
    if not text or not text.strip():
        return {
            "tokens": [], "sentences": [], "cleaned_tokens": [],
            "lemmas": [], "pos_tags": [], "named_entities": [],
            "vocab_size": 0, "top_words": []
        }

    # Sentences & Word Tokens
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Filtered tokens
    clean_words = [
        w.lower() for w in words
        if w.lower() not in STOPWORDS and w not in string.punctuation and not re.match(r"^\d+$", w)
    ]

    # Lemmatization & Stemming
    lemmas = [lemmatizer.lemmatize(w) for w in clean_words]
    stems = [stemmer.stem(w) for w in clean_words]

    # POS Tagging
    pos_tags = pos_tag(words)

    # NER via ne_chunk
    named_entities = []
    try:
        tree = ne_chunk(pos_tags)
        for chunk in tree:
            if hasattr(chunk, "label"):
                entity_name = " ".join(c[0] for c in chunk)
                named_entities.append(f"{entity_name} ({chunk.label()})")
    except Exception:
        pass

    # Frequency analysis
    freq = Counter(clean_words)
    top_words = freq.most_common(10)

    return {
        "sentence_count": len(sentences),
        "word_count": len(words),
        "clean_word_count": len(clean_words),
        "lexical_diversity": round(len(set(clean_words)) / max(1, len(clean_words)), 3),
        "sentences": sentences,
        "clean_tokens": clean_words,
        "lemmas": lemmas,
        "stems": stems,
        "pos_tags": pos_tags[:30],  # sample first 30 for display
        "named_entities": named_entities,
        "top_frequencies": top_words
    }
