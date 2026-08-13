"""
pipeline.py
-----------
Wires the four stages together: chunk -> embed -> retrieve -> generate.
This is the class that ingest.py and ask.py actually use.
"""

from pathlib import Path

from .chunking import chunk_documents
from .embeddings import get_embedder
from .vector_store import VectorStore
from .generator import generate_answer, build_prompt


class RAGPipeline:
    def __init__(self, embedding_backend: str = "tfidf"):
        self.embedder = get_embedder(embedding_backend)
        self.store = VectorStore()

    # ---------- ingestion (run once, or whenever your docs change) ----------

    def ingest(self, data_dir: str, chunk_size: int = 60, overlap: int = 15):
        docs = {}
        for path in sorted(Path(data_dir).glob("*.txt")):
            docs[path.name] = path.read_text(encoding="utf-8")

        if not docs:
            raise ValueError(f"No .txt files found in {data_dir}")

        chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
        texts = [c.text for c in chunks]

        self.embedder.fit(texts)
        vectors = self.embedder.embed(texts)
        self.store.add(chunks, vectors)

        return {"documents": len(docs), "chunks": len(chunks), "embedding_backend": self.embedder.name}

    def save(self, index_dir: str):
        self.store.save(index_dir)
        import pickle
        with open(Path(index_dir) / "embedder.pkl", "wb") as f:
            pickle.dump(self.embedder, f)

    @classmethod
    def load(cls, index_dir: str) -> "RAGPipeline":
        import pickle
        pipeline = cls.__new__(cls)
        pipeline.store = VectorStore.load(index_dir)
        with open(Path(index_dir) / "embedder.pkl", "rb") as f:
            pipeline.embedder = pickle.load(f)
        return pipeline

    # ---------- querying (run every time a user asks something) ----------

    def retrieve(self, question: str, top_k: int = 3):
        query_vector = self.embedder.embed([question])[0]
        return self.store.search(query_vector, top_k=top_k)

    def ask(self, question: str, top_k: int = 3, model: str = "claude-sonnet-4-6") -> dict:
        retrieved = self.retrieve(question, top_k=top_k)
        prompt = build_prompt(question, retrieved)
        answer = generate_answer(question, retrieved, model=model)
        return {
            "question": question,
            "retrieved": retrieved,
            "prompt": prompt,
            "answer": answer,
        }
