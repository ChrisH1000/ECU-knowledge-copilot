"""Ingest local documents into a persistent Chroma vector store.

This script loads documents from data/docs/, splits them into 1200-character
chunks with 120-character overlap (per PRD), embeds them, and stores the
vectors in Chroma for fast retrieval.

Usage:
    python ingest.py --rebuild  # Clear and rebuild the entire index
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retriever.factory import RetrieverConfig
from utils.embeddings import get_embeddings


# Use same collection name as retriever for consistency
COLLECTION_NAME = RetrieverConfig().collection_name

# Supported document types (in addition to PDFs)
SUPPORTED_TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """Parse command-line arguments for the ingestion script.

    Args:
        argv: Command-line arguments (typically sys.argv[1:])

    Returns:
        argparse.Namespace: Parsed arguments with docs, vectorstore, rebuild
    """
    parser = argparse.ArgumentParser(description="Build or update the local vector store.")
    parser.add_argument(
        "--docs",
        type=Path,
        default=None,
        help="Path to the source documents directory (overrides DOCS_PATH).",
    )
    parser.add_argument(
        "--vectorstore",
        type=Path,
        default=None,
        help="Destination directory for the Chroma store (overrides VECTORSTORE_PATH).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the vector store from scratch (clears existing data first).",
    )
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Determine source and destination paths from args and environment.

    Command-line arguments take precedence over environment variables,
    which in turn take precedence over hardcoded defaults.

    Args:
        args: Parsed command-line arguments

    Returns:
        tuple[Path, Path]: (docs_directory, vectorstore_directory)
    """
    docs_path = args.docs or Path(os.getenv("DOCS_PATH", "data/docs"))
    vector_path = args.vectorstore or Path(os.getenv("VECTORSTORE_PATH", "data/vectorstore"))
    return docs_path, vector_path


def load_documents(docs_root: Path) -> List[Document]:
    """Load all supported documents from the specified directory.

    Recursively scans for PDF, Markdown, and text files. Normalizes
    source paths to be relative to docs_root for cleaner citations.

    Args:
        docs_root: Root directory containing source documents

    Returns:
        list[Document]: Loaded documents with normalized metadata

    Raises:
        FileNotFoundError: If docs_root doesn't exist
        ValueError: If no supported documents are found
    """
    if not docs_root.exists():
        raise FileNotFoundError(f"Document directory not found: {docs_root}")

    documents: List[Document] = []

    # Recursively walk directory tree in sorted order
    for item in sorted(docs_root.rglob("*")):
        if item.is_dir():
            continue

        # Select appropriate loader based on file extension
        suffix = item.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(item))
        elif suffix in SUPPORTED_TEXT_EXTENSIONS:
            loader = TextLoader(str(item), autodetect_encoding=True)
        else:
            continue  # Skip unsupported file types

        # Load and normalize metadata for each document
        for doc in loader.load():
            source = doc.metadata.get("source") or str(item)
            try:
                # Convert absolute path to relative for cleaner citations
                relative = Path(source).relative_to(docs_root)
            except ValueError:
                # Fall back to original path if relative conversion fails
                relative = Path(source)
            doc.metadata["source"] = str(relative)
            documents.append(doc)

    if not documents:
        raise ValueError(f"No supported documents found under {docs_root}")

    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into fixed-size chunks with overlap.

    PRD specifies 1200-character chunks with 120-character overlap for
    optimal retrieval granularity.

    Args:
        documents: List of loaded documents

    Returns:
        list[Document]: Chunked documents ready for embedding
    """
    # Use recursive splitter for intelligent boundary detection
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
    return splitter.split_documents(documents)


def build_vector_store(chunks: List[Document], vector_path: Path) -> None:
    """Embed chunks and persist to Chroma vector store.

    Uses the configured embedding model (defaults to bge-small-en-v1.5)
    to convert text chunks into vectors and stores them in Chroma.

    Args:
        chunks: List of document chunks to embed
        vector_path: Directory where Chroma will persist the index
    """
    from langchain_chroma import Chroma

    # Initialize embedding model from environment (or use default)
    embedding_model = get_embeddings(
        model_name=os.getenv("EMBEDDING_MODEL"),
        device=os.getenv("EMBEDDING_DEVICE"),
    )

    # Create and persist vector store in one step
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(vector_path),
        collection_name=COLLECTION_NAME,
    )


def main(argv: Iterable[str]) -> int:
    """Orchestrate the full document ingestion pipeline.

    Steps:
    1. Load environment configuration
    2. Parse command-line arguments
    3. Optionally clear existing index (--rebuild)
    4. Load and chunk documents
    5. Embed and persist to Chroma

    Args:
        argv: Command-line arguments (typically sys.argv[1:])

    Returns:
        int: Exit code (0 for success)
    """
    # Load .env configuration
    load_dotenv()
    args = parse_args(argv)
    docs_path, vector_path = resolve_paths(args)

    # Clear existing index if --rebuild flag provided
    if args.rebuild and vector_path.exists():
        shutil.rmtree(vector_path)

    # Execute ingestion pipeline
    documents = load_documents(docs_path)
    chunks = chunk_documents(documents)
    build_vector_store(chunks, vector_path)

    # Report completion statistics
    print(f"Indexed {len(documents)} documents into {len(chunks)} chunks.")
    print(f"Vector store available at: {vector_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))