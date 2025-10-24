# Project: ECU Knowledge Copilot (Cost-Aware)

**Purpose**
Local RAG chatbot over internal PDFs/Markdown with minimal LLM cost.

## 1) Objectives
- Build a RAG pipeline that runs **entirely local** by default.
- Keep responses short, cite sources, and limit retrieval to the top 3 chunks.

## 2) Low-Cost Mode (defaults)
- **Models:** Ollama `llama3.1:8b-instruct` + embeddings `bge-small`.
- **Vector store:** Chroma.
- **Chunking:** 1200 chars, overlap 120; `k=3`.
- **Compression:** enable Maximal Marginal Relevance retriever for diversity.
- **Answer policy:** ≤ 120 words, 3 citations max.
- **Budget:** `max_calls=25` per session.

## 3) Functional Requirements
| Area | Description |
|------|-------------|
| Ingest | `ingest.py` loads `/data/docs/`, splits, embeds into Chroma |
| Query | `main.py` runs RetrievalQA with short, structured prompt |
| Memory | Lightweight buffer memory (cap 5 turns) |
| UI | CLI or Streamlit; show sources (file + page) |
| Export | Optional: save Q&A transcript to `logs/` |

## 4) Technical Stack
- Python 3.11; `langchain`, `chromadb`, `ollama`, `pypdf`, `unstructured`, `python-dotenv`, `streamlit` (optional).

## 5) Acceptance Criteria
- ✅ Runs without network access after initial setup.
- ✅ Answers include **concise text + citations**.
- ✅ Rebuild vector store with `python ingest.py --rebuild`.
- ✅ Respects BudgetManager and stops with a friendly message when exceeded.

## 6) Developer Notes

/ecu-knowledge-copilot/
├─ ingest.py              # loaders, splitters, embeddings
├─ main.py                # chat loop (CLI/Streamlit)
├─ retriever/
│   └─ factory.py         # MMR retriever, k=3
├─ prompts/
│   └─ answer.txt         # 120-word, cite-3 policy
├─ utils/
│   ├─ budget.py
│   └─ cache.py
└─ data/
├─ docs/
└─ vectorstore/

### Answer prompt (short)
- System: “Answer in ≤120 words. Cite up to 3 sources as [filename p.X]. If missing, say you don’t know.”
- User: “Question: {q}. Context: {top_chunks}.”