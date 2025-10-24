"""Command-line chat interface for the ECU Knowledge Copilot."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from retriever.factory import RetrieverConfig, load_vectorstore
from utils.budget import BudgetExceededError, BudgetManager
from utils.cache import ResponseCache
from utils.embeddings import get_embeddings


MAX_HISTORY_TURNS = 5


def resolve_model_configuration() -> tuple[str, str]:
    """Return the Ollama model configuration pulled from the environment."""

    model = os.getenv("OLLAMA_MODEL") or os.getenv("LLM_MODEL") or "llama3.1:8b-instruct"
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return model, base_url


def load_system_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_citation(metadata: Dict[str, object]) -> str:
    source = metadata.get("source", "unknown")
    filename = Path(str(source)).name
    page = metadata.get("page")
    if isinstance(page, (int, float)):
        page_index = int(page)
        if page_index >= 0:
            page_index += 1
        return f"[{filename} p.{page_index}]"
    section = metadata.get("section")
    if isinstance(section, str) and section:
        return f"[{filename} §{section}]"
    return f"[{filename}]"


def format_context(documents) -> Tuple[str, List[str]]:
    chunks: List[str] = []
    citations: List[str] = []
    for doc in documents[:3]:
        citation = build_citation(doc.metadata)
        if citation not in citations:
            citations.append(citation)
        chunks.append(f"{citation}\n{doc.page_content.strip()}")
    return "\n\n".join(chunks), citations


def enforce_word_limit(answer: str, limit: int = 120) -> str:
    words = answer.split()
    if len(words) <= limit:
        return answer.strip()
    return " ".join(words[:limit]).strip()


def ensure_citations(answer: str, citations: Sequence[str]) -> str:
    if not citations:
        return answer.strip()
    if any(citation in answer for citation in citations):
        return answer.strip()
    lines = [answer.strip(), "", "Sources:"]
    for citation in citations[:3]:
        lines.append(f"- {citation}")
    return "\n".join(lines)


def trim_history(history: List[BaseMessage]) -> List[BaseMessage]:
    limit = MAX_HISTORY_TURNS * 2
    if len(history) <= limit:
        return history
    return history[-limit:]


def should_exit(user_input: str) -> bool:
    return user_input.lower() in {"exit", "quit", "q"}


def write_transcript(log_dir: Path, transcript: List[Dict[str, object]]) -> None:
    if not transcript:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    destination = log_dir / f"session-{timestamp}.json"
    destination.write_text(json.dumps(transcript, indent=2), encoding="utf-8")


def build_chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("history"),
            (
                "human",
                "Question: {question}\n\n"
                "Context:\n{context}\n\n"
                "Available citations: {citations_string}",
            ),
        ]
    )

    model_name, base_url = resolve_model_configuration()
    llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.2)

    chain = prompt | llm | StrOutputParser()
    return chain


def main() -> int:
    load_dotenv()

    vector_path = Path(os.getenv("VECTORSTORE_PATH", "data/vectorstore"))
    if not vector_path.exists() or not any(vector_path.iterdir()):
        print("Vector store not found. Run `python ingest.py --rebuild` first.")
        return 1

    embedding = get_embeddings(
        model_name=os.getenv("EMBEDDING_MODEL"),
        device=os.getenv("EMBEDDING_DEVICE"),
    )
    _, retriever = load_vectorstore(vector_path, embedding, RetrieverConfig())

    system_prompt = load_system_prompt(Path("prompts/answer.txt"))
    chain = build_chain(system_prompt)

    budget = BudgetManager(max_calls=int(os.getenv("MAX_CALLS", "25")))
    cache = ResponseCache(Path(os.getenv("CACHE_PATH", "logs/cache.json")))

    history: List[BaseMessage] = []
    transcript: List[Dict[str, object]] = []

    print("ECU Knowledge Copilot (type 'exit' to quit)")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if should_exit(question):
            break

        documents = retriever.invoke(question)
        if not documents:
            answer_text = "I don't know based on the provided documents."
            citations: List[str] = []
        else:
            context, citations = format_context(documents)
            signature = {
                "question": question,
                "citations": citations,
            }
            cached = cache.get(signature)

            if cached:
                answer_text = cached["answer"]
            else:
                try:
                    budget.register_call()
                except BudgetExceededError as exc:
                    print(exc)
                    break

                answer_text = chain.invoke(
                    {
                        "question": question,
                        "context": context,
                        "citations_string": ", ".join(citations),
                        "history": history,
                    }
                )
                answer_text = enforce_word_limit(answer_text)
                answer_text = ensure_citations(answer_text, citations)
                cache.set(signature, {"answer": answer_text, "citations": citations})

        print(f"Copilot: {answer_text}\n")

        history.extend([HumanMessage(content=question), AIMessage(content=answer_text)])
        history = trim_history(history)

        transcript.append(
            {
                "question": question,
                "answer": answer_text,
                "citations": citations,
            }
        )

    write_transcript(Path("logs"), transcript)
    return 0


if __name__ == "__main__":
    sys.exit(main())