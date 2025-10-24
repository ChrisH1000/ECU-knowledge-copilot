# ECU Knowledge Copilot

Local retrieval-augmented generation (RAG) chatbot designed to answer questions about ECU operational knowledge while keeping language-model usage inexpensive.

## Key Features
- Runs entirely on local infrastructure using [Ollama](https://ollama.ai/) chat models and Hugging Face embeddings.
- Cost-aware design with a session budget limiter (25 calls by default) and persistent response cache.
- Citation-first answers capped at 120 words, with up to three sources pulled via Maximal Marginal Relevance (MMR) retrieval.
- Rebuildable document index backed by Chroma for fast similarity search.
- Transcript logging for every CLI conversation in `logs/`.

## Prerequisites
- **Python** 3.11 (recommended for compatibility with the LangChain ecosystem).
- **Ollama** installed and running locally. Follow the [LangChain ChatOllama setup guide](https://python.langchain.com/docs/integrations/chat/ollama/) for installation details (macOS users can `brew install ollama` and start it with `brew services start ollama`).
- At least one document placed in `data/docs/` (Markdown, text, or PDF).

## Installation
```bash
git clone <repository-url>
cd "ECU knowledge copilot"

python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt

# Pull the default Ollama model referenced by the CLI
ollama pull llama3.1:8b-instruct
```

> **Tip:** The embeddings default to `BAAI/bge-small-en-v1.5` and download automatically from Hugging Face when first used.

## Environment Configuration
Create a `.env` file (or copy `.env.example`) and adjust as needed:

```ini
OLLAMA_MODEL=llama3.1:8b-instruct
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=cpu
VECTORSTORE_PATH=data/vectorstore
DOCS_PATH=data/docs
MAX_CALLS=25
CACHE_PATH=logs/cache.json
```

Only `OLLAMA_MODEL` and `OLLAMA_BASE_URL` are required if you stick with the defaults. Setting `EMBEDDING_DEVICE` to `cuda` (where available) will speed up ingestion and queries.

## Indexing Documents
Before the chatbot can answer questions you must ingest the source material:

```bash
python ingest.py --rebuild
```

This command loads every supported file in `data/docs/`, splits the text into 1,200-character chunks (120-character overlap), embeds them with task-specific prefixes, and stores the vectors in Chroma at `data/vectorstore/`.

## Running the CLI
Start the copilot with:

```bash
python main.py
```

You will see a prompt like `ECU Knowledge Copilot (type 'exit' to quit)`. Ask questions freely; each answer will:
- Remain within 120 words.
- Include up to three citations (e.g., `[sample.md]`).
- Stop responding once the 25-call budget is reached.

Type `exit`, `quit`, or press `Ctrl+C` to end the session. A JSON transcript is saved to `logs/session-<timestamp>.json`.

## Testing
Run the unit suite to verify helper utilities (budget manager, cache logic, etc.):

```bash
python -m pytest
```

## Project Layout
```
├─ ingest.py            # Document loaders, splitter, embedding configuration
├─ main.py              # CLI chat loop with budget and caching controls
├─ prompts/answer.txt   # Prompt enforcing word and citation limits
├─ retriever/factory.py # Chroma retriever factory (MMR, k=3)
├─ utils/
│   ├─ budget.py        # Session budget enforcement
│   ├─ cache.py         # Simple JSON cache for model responses
│   └─ embeddings.py    # Instruction-aware Hugging Face embeddings
├─ data/
│   ├─ docs/            # Source material to ingest
│   └─ vectorstore/     # Persisted Chroma index
└─ tests/               # Pytest-based regression coverage
```

## Troubleshooting
- **Vector store missing:** Run `python ingest.py --rebuild` after adding or changing documents.
- **Ollama connection errors:** Ensure the Ollama daemon is running (`ollama serve` or `brew services start ollama`) and that the base URL matches your `.env`.
- **Slow embedding downloads:** The first ingest may download model artifacts; subsequent runs reuse the cached models. Setting `HF_HOME` can relocate the cache if needed.
- **SSL warning (macOS + Python 3.9):** `urllib3` may emit a `NotOpenSSLWarning` when Python links against LibreSSL. This warning is benign for local workflows.

## References
- LangChain documentation for [ChatOllama integration](https://python.langchain.com/docs/integrations/chat/ollama) (accessed October 2025).
