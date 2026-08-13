"""
test_pipeline.py
-----------------
Fast, offline tests using the tfidf backend - no API key or model
downloads needed, so these can run in CI (see .github/workflows).

Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking import chunk_text, chunk_documents
from src.pipeline import RAGPipeline


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, source="test.txt", chunk_size=20, overlap=5)

    assert len(chunks) > 1
    # Each chunk (except possibly the last) should have chunk_size words
    assert len(chunks[0].text.split()) == 20
    # Consecutive chunks should share `overlap` words
    first_tail = chunks[0].text.split()[-5:]
    second_head = chunks[1].text.split()[:5]
    assert first_tail == second_head


def test_chunk_documents_tags_source_correctly():
    docs = {"a.txt": "one two three four five six", "b.txt": "seven eight nine ten"}
    chunks = chunk_documents(docs, chunk_size=3, overlap=1)
    sources = {c.source for c in chunks}
    assert sources == {"a.txt", "b.txt"}


def test_pipeline_retrieves_relevant_document():
    pipeline = RAGPipeline(embedding_backend="tfidf")
    pipeline.ingest("data")

    results = pipeline.retrieve("What does chlorophyll do in plants?", top_k=3)
    top_sources = [chunk.source for chunk, score in results[:2]]

    assert "photosynthesis.txt" in top_sources


def test_pipeline_low_confidence_on_unrelated_query():
    pipeline = RAGPipeline(embedding_backend="tfidf")
    pipeline.ingest("data")

    results = pipeline.retrieve("What is the capital of Peru?", top_k=3)
    top_score = results[0][1]

    # Nothing in our corpus is about world capitals, so the best match
    # should have very low similarity - this is the signal a real system
    # should check before trusting the retrieved context.
    assert top_score < 0.15


def test_save_and_load_index_roundtrip(tmp_path):
    pipeline = RAGPipeline(embedding_backend="tfidf")
    pipeline.ingest("data")
    pipeline.save(str(tmp_path / "index"))

    reloaded = RAGPipeline.load(str(tmp_path / "index"))
    results = reloaded.retrieve("How do vaccines train the immune system?", top_k=1)

    assert results[0][0].source == "vaccines.txt"
