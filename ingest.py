"""
ingest.py
---------
Step 1 of using this project: build the vector index from data/*.txt.

Usage:
    python ingest.py
    python ingest.py --backend sentence-transformers
    python ingest.py --data-dir data --index-dir index --chunk-size 60 --overlap 15
"""

import argparse

from dotenv import load_dotenv

from src.pipeline import RAGPipeline

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Build a RAG vector index from a folder of .txt files.")
    parser.add_argument("--data-dir", default="data", help="Folder of .txt documents to ingest")
    parser.add_argument("--index-dir", default="index", help="Where to save the built index")
    parser.add_argument("--backend", default="tfidf", choices=["tfidf", "sentence-transformers"],
                         help="Embedding backend. tfidf works offline; sentence-transformers is semantic but downloads a model.")
    parser.add_argument("--chunk-size", type=int, default=60, help="Words per chunk")
    parser.add_argument("--overlap", type=int, default=15, help="Overlapping words between consecutive chunks")
    args = parser.parse_args()

    print(f"Ingesting documents from '{args.data_dir}/' using the '{args.backend}' embedding backend...")

    pipeline = RAGPipeline(embedding_backend=args.backend)
    stats = pipeline.ingest(args.data_dir, chunk_size=args.chunk_size, overlap=args.overlap)
    pipeline.save(args.index_dir)

    print(f"Done.")
    print(f"  Documents ingested : {stats['documents']}")
    print(f"  Chunks created     : {stats['chunks']}")
    print(f"  Embedding backend  : {stats['embedding_backend']}")
    print(f"  Index saved to     : {args.index_dir}/")
    print(f"\nNext step: python ask.py \"your question here\"")


if __name__ == "__main__":
    main()
