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
    """Return the Ollama model configuration pulled from the environment.

    Checks OLLAMA_MODEL first, falls back to LLM_MODEL, then to the default
    llama3.1:8b-instruct model. Base URL defaults to localhost:11434.

    Returns:
        tuple[str, str]: (model_name, base_url) for ChatOllama initialization
    """
    # Try environment variables in order of preference, fall back to default
    model = os.getenv("OLLAMA_MODEL") or os.getenv("LLM_MODEL") or "llama3.1:8b-instruct"
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return model, base_url


def load_system_prompt(path: Path) -> str:
    """Load the system prompt template from disk.

    Args:
        path: Path to the prompt text file

    Returns:
        str: Stripped prompt content

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_citation(metadata: Dict[str, object]) -> str:
    """Format a citation string from document metadata.

    Builds citations in the format [filename p.X] for PDFs (1-indexed page numbers),
    [filename §section] for documents with section metadata, or just [filename]
    as a fallback.

    Args:
        metadata: Document metadata dictionary containing source and optional page/section

    Returns:
        str: Formatted citation string
    """
    # Extract source path and convert to filename
    source = metadata.get("source", "unknown")
    filename = Path(str(source)).name

    # Try page-based citation (common for PDFs)
    page = metadata.get("page")
    if isinstance(page, (int, float)):
        page_index = int(page)
        # Convert 0-indexed to 1-indexed for human readability
        if page_index >= 0:
            page_index += 1
        return f"[{filename} p.{page_index}]"

    # Try section-based citation (for structured documents)
    section = metadata.get("section")
    if isinstance(section, str) and section:
        return f"[{filename} §{section}]"

    # Fallback: filename only
    return f"[{filename}]"


def format_context(documents) -> Tuple[str, List[str]]:
    """Format retrieved documents into context string and citation list.

    Limits to the top 3 documents (as specified by PRD), builds citations
    for each, and formats chunks with their citations prepended.

    Args:
        documents: List of retrieved Document objects from the vector store

    Returns:
        tuple[str, list[str]]: (formatted_context, unique_citations)
    """
    chunks: List[str] = []
    citations: List[str] = []

    # Process only the top 3 documents (PRD requirement)
    for doc in documents[:3]:
        # Build and track unique citations
        citation = build_citation(doc.metadata)
        if citation not in citations:
            citations.append(citation)

        # Format each chunk with its citation header
        chunks.append(f"{citation}\n{doc.page_content.strip()}")

    # Join chunks with double newlines for readability
    return "\n\n".join(chunks), citations


def enforce_word_limit(answer: str, limit: int = 120) -> str:
    """Truncate answer to specified word count.

    PRD specifies answers should be ≤120 words. This function enforces
    that limit by splitting on whitespace and rejoining.

    Args:
        answer: The model's raw response text
        limit: Maximum number of words (default 120)

    Returns:
        str: Truncated answer if over limit, original if under
    """
    words = answer.split()
    if len(words) <= limit:
        return answer.strip()
    # Take only the first 'limit' words
    return " ".join(words[:limit]).strip()


def ensure_citations(answer: str, citations: Sequence[str]) -> str:
    """Append citations to answer if not already present.

    Checks whether the model included citations in its response. If not,
    appends them in a "Sources:" section. Limits to 3 citations per PRD.

    Args:
        answer: The answer text (possibly already containing citations)
        citations: List of citation strings to include

    Returns:
        str: Answer with citations guaranteed to be present
    """
    # No citations to add
    if not citations:
        return answer.strip()

    # Model already included citations, don't duplicate
    if any(citation in answer for citation in citations):
        return answer.strip()

    # Append citations in a structured format
    lines = [answer.strip(), "", "Sources:"]
    for citation in citations[:3]:  # Limit to 3 per PRD
        lines.append(f"- {citation}")
    return "\n".join(lines)


def trim_history(history: List[BaseMessage]) -> List[BaseMessage]:
    """Keep only the most recent conversation turns.

    PRD specifies a 5-turn memory cap. Since each turn consists of a
    HumanMessage and AIMessage, we keep the last 10 messages.

    Args:
        history: Full conversation history as a list of messages

    Returns:
        list[BaseMessage]: Trimmed history (most recent turns only)
    """
    limit = MAX_HISTORY_TURNS * 2  # Human + AI per turn
    if len(history) <= limit:
        return history
    # Keep only the most recent messages
    return history[-limit:]


def should_exit(user_input: str) -> bool:
    """Check if user input signals intent to quit.

    Args:
        user_input: Raw user input string

    Returns:
        bool: True if input matches exit commands
    """
    return user_input.lower() in {"exit", "quit", "q"}


def write_transcript(log_dir: Path, transcript: List[Dict[str, object]]) -> None:
    """Save conversation transcript to a timestamped JSON file.

    Creates the log directory if needed, then writes the full session
    transcript with questions, answers, and citations.

    Args:
        log_dir: Directory to save transcripts
        transcript: List of Q&A dictionaries to serialize
    """
    # Skip if no conversation occurred
    if not transcript:
        return

    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped filename
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    destination = log_dir / f"session-{timestamp}.json"

    # Write formatted JSON
    destination.write_text(json.dumps(transcript, indent=2), encoding="utf-8")


def build_chain(system_prompt: str):
    """Construct the LangChain LCEL pipeline for RAG question-answering.

    Combines a prompt template (with system instructions, conversation history,
    and retrieved context) with the Ollama chat model and a string parser.

    Args:
        system_prompt: The system-level instruction text from prompts/answer.txt

    Returns:
        Runnable: LangChain Expression Language chain ready for invocation
    """
    # Build prompt template with system message, chat history, and user query
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),  # Instructions for word limit and citations
            MessagesPlaceholder("history"),  # Conversation memory (5 turns)
            (
                "human",
                "Question: {question}\n\n"
                "Context:\n{context}\n\n"
                "Available citations: {citations_string}",
            ),
        ]
    )

    # Initialize Ollama chat model with environment-specified configuration
    model_name, base_url = resolve_model_configuration()
    llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.2)

    # Compose LCEL chain: prompt -> model -> string output
    chain = prompt | llm | StrOutputParser()
    return chain


def main() -> int:
    """Run the CLI chatbot with RAG retrieval, budget enforcement, and caching.

    Orchestrates the full question-answering loop:
    1. Load environment configuration and validate vector store exists
    2. Initialize embeddings, retriever, and LLM chain
    3. Set up session budget (25 calls default) and response cache
    4. Loop: accept user questions, retrieve context, generate answers
    5. Save transcript on exit

    Returns:
        int: Exit code (0 for success, 1 for missing vector store)
    """
    # Load environment variables from .env file
    load_dotenv()

    # Validate that the vector store has been created
    vector_path = Path(os.getenv("VECTORSTORE_PATH", "data/vectorstore"))
    if not vector_path.exists() or not any(vector_path.iterdir()):
        print("Vector store not found. Run `python ingest.py --rebuild` first.")
        return 1

    # Initialize embedding model (defaults to bge-small-en-v1.5)
    embedding = get_embeddings(
        model_name=os.getenv("EMBEDDING_MODEL"),
        device=os.getenv("EMBEDDING_DEVICE"),
    )

    # Load Chroma vector store and create MMR retriever (k=3)
    _, retriever = load_vectorstore(vector_path, embedding, RetrieverConfig())

    # Load system prompt and build the LangChain LCEL pipeline
    system_prompt = load_system_prompt(Path("prompts/answer.txt"))
    chain = build_chain(system_prompt)

    # Initialize session budget manager (stops after 25 LLM calls)
    budget = BudgetManager(max_calls=int(os.getenv("MAX_CALLS", "25")))

    # Initialize response cache to avoid redundant LLM calls
    cache = ResponseCache(Path(os.getenv("CACHE_PATH", "logs/cache.json")))

    # Track conversation history (trimmed to 5 turns) and full transcript
    history: List[BaseMessage] = []
    transcript: List[Dict[str, object]] = []

    print("ECU Knowledge Copilot (type 'exit' to quit)")

    # Main conversation loop
    while True:
        # Accept user input with graceful handling of interrupts
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        # Skip empty inputs
        if not question:
            continue

        # Check for exit commands
        if should_exit(question):
            break

        # Retrieve relevant document chunks using MMR (up to 3)
        documents = retriever.invoke(question)

        if not documents:
            # No relevant context found
            answer_text = "I don't know based on the provided documents."
            citations: List[str] = []
        else:
            # Format retrieved documents into context and citation list
            context, citations = format_context(documents)

            # Create cache signature based on question and retrieved sources
            signature = {
                "question": question,
                "citations": citations,
            }
            cached = cache.get(signature)

            if cached:
                # Use cached response to save LLM calls
                answer_text = cached["answer"]
            else:
                # Generate new answer via LLM chain
                try:
                    # Enforce session budget before making LLM call
                    budget.register_call()
                except BudgetExceededError as exc:
                    # Stop session if budget exceeded
                    print(exc)
                    break

                # Invoke the RAG chain with question, context, and history
                answer_text = chain.invoke(
                    {
                        "question": question,
                        "context": context,
                        "citations_string": ", ".join(citations),
                        "history": history,
                    }
                )

                # Apply PRD constraints: 120-word limit and ensure citations
                answer_text = enforce_word_limit(answer_text)
                answer_text = ensure_citations(answer_text, citations)

                # Cache the generated answer
                cache.set(signature, {"answer": answer_text, "citations": citations})

        # Display answer to user
        print(f"Copilot: {answer_text}\n")

        # Update conversation history with this turn
        history.extend([HumanMessage(content=question), AIMessage(content=answer_text)])
        history = trim_history(history)

        # Record Q&A in transcript
        transcript.append(
            {
                "question": question,
                "answer": answer_text,
                "citations": citations,
            }
        )

    # Save conversation transcript to logs/ on exit
    write_transcript(Path("logs"), transcript)
    return 0


if __name__ == "__main__":
    sys.exit(main())