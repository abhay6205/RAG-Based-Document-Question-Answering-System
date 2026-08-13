"""
ask.py
------
Step 2 of using this project: ask questions against the index built by ingest.py.

Usage:
    python ask.py "How does photosynthesis work?"
    python ask.py "What ratio of coffee to water should I use?" --top-k 2
    python ask.py "What's the capital of Peru?"   # tests what happens with no relevant context
    python ask.py "..." --no-generate             # retrieval only, no API call/key needed
"""

import argparse

from dotenv import load_dotenv

from src.pipeline import RAGPipeline

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Ask a question against a built RAG index.")
    parser.add_argument("question", help="Your question, in quotes")
    parser.add_argument("--index-dir", default="index", help="Where the index was saved by ingest.py")
    parser.add_argument("--top-k", type=int, default=3, help="How many chunks to retrieve")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Claude model to use for generation")
    parser.add_argument("--no-generate", action="store_true",
                         help="Only run retrieval and print the chunks/prompt - skips the API call. "
                              "Useful for testing without an ANTHROPIC_API_KEY.")
    args = parser.parse_args()

    pipeline = RAGPipeline.load(args.index_dir)

    retrieved = pipeline.retrieve(args.question, top_k=args.top_k)

    print(f"\nQuestion: {args.question}")
    print(f"\nTop {args.top_k} retrieved chunks:")
    for i, (chunk, score) in enumerate(retrieved, 1):
        preview = chunk.text[:140] + ("..." if len(chunk.text) > 140 else "")
        print(f"  {i}. [{chunk.source} | score={score:.3f}] {preview}")

    if args.no_generate:
        from src.generator import build_prompt
        print("\n--no-generate set, skipping the LLM call. Here's the prompt that WOULD be sent:\n")
        print(build_prompt(args.question, retrieved))
        return

    print("\nGenerating answer...")
    try:
        result = pipeline.ask(args.question, top_k=args.top_k, model=args.model)
    except RuntimeError as e:
        print(f"\nCouldn't generate an answer: {e}")
        return
    print(f"\nAnswer:\n{result['answer']}\n")


if __name__ == "__main__":
    main()
