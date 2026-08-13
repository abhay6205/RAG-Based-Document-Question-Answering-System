"""
embeddings.py
-------------
Turns text into vectors so we can compare "meaning" numerically.

Two backends are provided, on purpose, so you can see the difference:

1. TfidfEmbedder
   - Classic, no downloads, works fully offline.
   - Represents text as weighted word-overlap. It matches VOCABULARY, not
     meaning: "car" and "automobile" look unrelated to it.
   - Good for learning the mechanics and for quick local testing.

2. SentenceTransformerEmbedder
   - A real neural embedding model (all-MiniLM-L6-v2) that captures
     semantic meaning, not just word overlap. "car" and "automobile" end
     up close together in vector space.
   - Downloads a small (~80MB) model from Hugging Face on first run, so it
     needs internet access the first time.
   - This is what production RAG systems actually use (or a paid API
     equivalent like OpenAI/Cohere/Voyage embeddings).

Both expose the same interface: fit(texts) then embed(texts) -> np.ndarray,
so the rest of the pipeline doesn't care which one you're using.
"""

import numpy as np


class TfidfEmbedder:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False

    def fit(self, texts: list[str]):
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit(texts) before embed(). TF-IDF needs a vocabulary first.")
        return self.vectorizer.transform(texts).toarray()

    @property
    def name(self):
        return "tfidf"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]):
        # Neural embedders don't need a separate fit step over the corpus -
        # the model was already pretrained. Kept here only so the interface
        # matches TfidfEmbedder and the pipeline code doesn't need branches.
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, show_progress_bar=False))

    @property
    def name(self):
        return "sentence-transformers/all-MiniLM-L6-v2"


def get_embedder(backend: str = "tfidf"):
    if backend == "tfidf":
        return TfidfEmbedder()
    elif backend == "sentence-transformers":
        return SentenceTransformerEmbedder()
    else:
        raise ValueError(f"Unknown embedding backend: {backend!r}. Use 'tfidf' or 'sentence-transformers'.")
