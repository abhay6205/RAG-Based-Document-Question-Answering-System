"""
vector_store.py
----------------
The simplest possible vector database: hold all chunk vectors in a numpy
array in memory, and find the closest ones to a query vector by cosine
similarity.

Real production vector databases (FAISS, Pinecone, Weaviate, Chroma, etc.)
do the same core math, but add approximate nearest-neighbor indexing so
this stays fast at millions of vectors instead of thousands. For a
learning project - or anything up to tens of thousands of chunks - brute
force cosine similarity like this is genuinely fine and a lot easier to
understand.
"""

import json
import pickle
from pathlib import Path

import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray):
        assert len(chunks) == vectors.shape[0], "Number of chunks must match number of vectors"
        self.chunks = chunks
        self.vectors = vectors

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[tuple[Chunk, float]]:
        """
        Returns the top_k (chunk, similarity_score) pairs, sorted highest
        similarity first. Similarity is cosine similarity, ranging from
        -1 (opposite) to 1 (identical), though in practice for text it's
        almost always between 0 and 1.
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []

        query_vector = query_vector.reshape(1, -1)

        # cosine similarity = dot(a, b) / (||a|| * ||b||)
        dot = self.vectors @ query_vector.T  # shape: (n_chunks, 1)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) * np.linalg.norm(query_vector)
        norms[norms == 0] = 1e-9  # avoid divide-by-zero for empty vectors
        similarities = (dot / norms).flatten()

        top_indices = np.argsort(-similarities)[:top_k]
        return [(self.chunks[i], float(similarities[i])) for i in top_indices]

    def save(self, path: str):
        """Persist the index to disk so ingest.py and ask.py can run as separate steps."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        meta = {"n_chunks": len(self.chunks), "vector_dim": int(self.vectors.shape[1])}
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        path = Path(path)
        store = cls()
        store.vectors = np.load(path / "vectors.npy")
        with open(path / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store
