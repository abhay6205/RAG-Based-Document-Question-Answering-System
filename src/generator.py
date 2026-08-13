"""
generator.py
------------
The "G" in RAG. Takes the user's question plus the retrieved chunks and
asks an LLM to answer using that context.

This is deliberately a thin wrapper - the interesting part of RAG is the
retrieval above it, not the API call itself.
"""

import os

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.

Rules:
- If the answer is fully contained in the context, answer it clearly and cite which source it came from.
- If the context only partially answers the question, say what's missing.
- If the context does not contain the answer at all, say so explicitly instead of guessing.
- Do not use outside knowledge, even if you know the answer."""


def build_prompt(question: str, retrieved: list[tuple]) -> str:
    """
    retrieved: list of (Chunk, score) tuples, typically from VectorStore.search()
    """
    context_blocks = []
    for chunk, score in retrieved:
        context_blocks.append(f"[Source: {chunk.source} | relevance: {score:.3f}]\n{chunk.text}")
    context = "\n\n---\n\n".join(context_blocks)

    return f"""Context:
{context}

Question: {question}"""


def generate_answer(question: str, retrieved: list[tuple], model: str = "claude-sonnet-4-6") -> str:
    """
    Calls the Anthropic API with the retrieved context. Requires
    ANTHROPIC_API_KEY to be set in the environment (see .env.example).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env, add your key, "
            "and make sure it's loaded (ingest.py/ask.py already call load_dotenv())."
        )

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    prompt = build_prompt(question, retrieved)
    response = client.messages.create(
        model=model,
        max_tokens=500,
        system=RAG_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
