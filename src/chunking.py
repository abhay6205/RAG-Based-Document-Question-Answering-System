"""
chunking.py
-----------
Splits raw documents into smaller overlapping "chunks" before embedding.

Why chunk at all?
- Embedding models and retrieval work better on focused pieces of text than
  on whole documents (a 2000-word doc "about" ten things retrieves poorly
  for a question about one of those ten things).
- The LLM's context window is limited, so we want to hand it only the
  most relevant few hundred words, not entire files.

Why overlap chunks?
- If a sentence that answers the question sits right at a chunk boundary,
  a hard cut can split it across two chunks and weaken both. A small
  overlap (e.g. 20% of chunk size) makes that much less likely.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str        # which file/document this chunk came from
    chunk_id: int       # position of this chunk within that document


def chunk_text(text: str, source: str, chunk_size: int = 60, overlap: int = 15) -> list[Chunk]:
    """
    Split `text` into word-based chunks.

    chunk_size: number of words per chunk
    overlap: number of words shared between consecutive chunks
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    chunk_id = 0
    step = max(1, chunk_size - overlap)

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(Chunk(
            text=" ".join(chunk_words),
            source=source,
            chunk_id=chunk_id,
        ))
        chunk_id += 1
        if end == len(words):
            break
        start += step

    return chunks


def chunk_documents(docs: dict[str, str], chunk_size: int = 60, overlap: int = 15) -> list[Chunk]:
    """
    docs: mapping of {filename: raw_text}
    Returns a flat list of Chunk objects across all documents.
    """
    all_chunks = []
    for source, text in docs.items():
        all_chunks.extend(chunk_text(text, source, chunk_size, overlap))
    return all_chunks
