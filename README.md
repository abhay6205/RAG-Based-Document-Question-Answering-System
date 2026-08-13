# RAG from Scratch

A minimal, fully working Retrieval-Augmented Generation (RAG) pipeline, built to be read and understood end to end — not just run. No LangChain, no vector DB service, no hidden magic. Every stage (chunking, embedding, retrieval, generation) is ~50 lines of plain Python you can open and read.

```
Question -> embed -> search vector store -> top-k chunks -> build prompt -> Claude -> grounded answer
```

## What's actually in here

```
rag-project/
├── data/                  # 6 sample .txt documents (the knowledge base)
├── src/
│   ├── chunking.py        # splits documents into overlapping word chunks
│   ├── embeddings.py      # text -> vectors (TF-IDF or sentence-transformers)
│   ├── vector_store.py    # stores vectors, does cosine-similarity search
│   ├── generator.py       # builds the augmented prompt, calls Claude
│   └── pipeline.py        # wires the above into ingest() / ask()
├── ingest.py               # CLI: build the index from data/
├── ask.py                  # CLI: ask a question against the index
├── tests/test_pipeline.py  # offline tests, no API key needed
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

```bash
git clone <your-repo-url>
cd rag-project

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# now open .env and paste your key from https://console.anthropic.com/settings/keys
```

## Step-by-step execution

### 1. Build the index

```bash
python ingest.py
```

This reads every `.txt` file in `data/`, splits each into overlapping ~60-word chunks, converts each chunk into a vector, and saves everything to `index/`. You'll see:

```
Ingesting documents from 'data/' using the 'tfidf' embedding backend...
Done.
  Documents ingested : 6
  Chunks created      : 19
  Embedding backend   : tfidf
  Index saved to      : index/
```

You only need to re-run this when your documents change — it's a separate step from asking questions on purpose, exactly like a real system where ingestion is a batch job and querying is a live request.

### 2. Try retrieval alone first (no API key needed)

Before spending an API call, look at what gets retrieved:

```bash
python ask.py "Why do leaves look green?" --no-generate
```

```
Top 3 retrieved chunks:
  1. [photosynthesis.txt | score=0.294] Photosynthesis is the process plants use...
  2. [photosynthesis.txt | score=0.161] most plants appear green. The overall reaction...
  3. [black_holes.txt | score=0.000] A black hole forms when a sufficiently massive star...
```

Notice the third result scored exactly `0.000` — that's the top-k slot getting backfilled with something irrelevant because nothing else scored higher. This is the single most important number in any RAG system: **retrieval confidence**. A production system should check this before generating and refuse to answer (or say "I don't know") when even the best match is near zero, rather than handing the LLM garbage context and letting it guess anyway.

Try it on a question totally outside the corpus:

```bash
python ask.py "What's the capital of Peru?" --no-generate
```

Every score comes back `~0.000`. Good — that's the system correctly telling you it has nothing relevant, before you've spent a single token on generation.

### 3. Ask for real

```bash
python ask.py "Why do leaves look green?"
```

This retrieves the same chunks as above, wraps them plus your question in a prompt (view it any time with `--no-generate`), and sends it to Claude with a system prompt that instructs it to answer **only** from the provided context and say so explicitly if the context doesn't cover the question. You'll get back an answer grounded in `photosynthesis.txt`, with no invented facts.

Compare that to asking Claude the same question with zero context — it'll probably still get it right (this is common knowledge), but on your own private/niche documents, the difference between grounded and ungrounded answers becomes obvious fast. Try adding your own `.txt` file to `data/`, re-running `ingest.py`, and asking about something only that file knows.

### 4. Run the tests

```bash
python -m pytest tests/ -v
```

Five tests, all offline (TF-IDF backend, no API key or downloads required):
- chunking produces correctly sized, correctly overlapping chunks
- chunks are tagged with the right source document
- a relevant query retrieves the right document
- an unrelated query produces near-zero similarity scores
- save/load round-trips an index correctly

## A deliberate limitation, worth understanding

Try this:

```bash
python ask.py "How does photosynthesis work?" --no-generate
```

You might notice a `vaccines.txt` chunk outscoring a `photosynthesis.txt` chunk. That's not a bug — it's TF-IDF being exactly what it is: **word-overlap matching, not meaning matching**. The vaccines document literally contains the word "work" ("Vaccines *work* by training..."), while the best photosynthesis chunk uses "process" instead. TF-IDF has no idea those are related concepts.

This is the single biggest limitation of the default setup, and it's why real systems use neural embeddings instead.

### Upgrading to real semantic embeddings

```bash
pip install sentence-transformers
python ingest.py --backend sentence-transformers
python ask.py "How does photosynthesis work?" --no-generate
```

This swaps in `all-MiniLM-L6-v2`, a small neural embedding model (downloaded once, ~80MB, from Hugging Face) that captures actual meaning rather than word overlap. Re-run the same query and watch the ranking correct itself. Because `embeddings.py` is written behind one small interface (`fit` / `embed`), nothing else in the project has to change to support this — that's the point of keeping retrieval and generation as separate, swappable pieces.

## How each stage works (short version)

| Stage | File | What it does |
|---|---|---|
| **Chunk** | `chunking.py` | Splits raw text into ~60-word pieces with 15-word overlap, so an answer straddling a chunk boundary doesn't get cut in half. |
| **Embed** | `embeddings.py` | Converts each chunk (and later, each query) into a vector. TF-IDF: word-overlap vectors, offline. Sentence-transformers: dense semantic vectors, downloads a model once. |
| **Store & retrieve** | `vector_store.py` | Holds all chunk vectors in memory; ranks them against a query vector by cosine similarity — the same math a "real" vector database like FAISS or Pinecone uses under the hood, just without approximate-nearest-neighbor indexing for scale. |
| **Generate** | `generator.py` | Builds the final prompt (context + question) and calls Claude with a system prompt that enforces answering only from the given context. |

## Extending this project

Ideas if you want to keep building, roughly in order of effort:

- **Bigger corpus**: point `--data-dir` at a folder of your own notes, docs, or a wiki export.
- **PDF support**: add a loader in `ingest.py` using `pypdf` to extract text before chunking.
- **Confidence gating**: in `ask.py`, skip generation entirely (or return "I don't know") when the top retrieval score is below a threshold — you already have the number, just add the `if`.
- **Hybrid search**: combine TF-IDF (good at exact keyword/name matches) with semantic embeddings (good at meaning) and merge the rankings.
- **Reranking**: retrieve top 10 with the fast method, then re-score just those 10 with a slower, more accurate model before picking the final top-3.
- **A real vector database**: swap `vector_store.py`'s numpy array for FAISS (`faiss-cpu`) once you're past a few thousand chunks and brute-force search gets slow.
- **A UI**: wrap `RAGPipeline` in a small Streamlit or Flask app instead of the CLI.

## Publishing to GitHub

```bash
cd rag-project
git init
git add .
git commit -m "Initial commit: working RAG pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

`.gitignore` already excludes `.env` (your API key) and `index/` (regenerable build output), so neither gets committed. Anyone who clones the repo just needs to run `pip install -r requirements.txt`, add their own `.env`, and run `python ingest.py` to reproduce your index from scratch.

## License

MIT — do whatever you want with this.
