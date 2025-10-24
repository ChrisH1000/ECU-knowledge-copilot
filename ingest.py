"""Ingest local documents into a persistent Chroma vector store."""

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


COLLECTION_NAME = RetrieverConfig().collection_name
SUPPORTED_TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
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
    docs_path = args.docs or Path(os.getenv("DOCS_PATH", "data/docs"))
    vector_path = args.vectorstore or Path(os.getenv("VECTORSTORE_PATH", "data/vectorstore"))
    return docs_path, vector_path


def load_documents(docs_root: Path) -> List[Document]:
    if not docs_root.exists():
        raise FileNotFoundError(f"Document directory not found: {docs_root}")

    documents: List[Document] = []
    for item in sorted(docs_root.rglob("*")):
        if item.is_dir():
            continue
        suffix = item.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(item))
        elif suffix in SUPPORTED_TEXT_EXTENSIONS:
            loader = TextLoader(str(item), autodetect_encoding=True)
        else:
            continue

        for doc in loader.load():
            source = doc.metadata.get("source") or str(item)
            try:
                relative = Path(source).relative_to(docs_root)
            except ValueError:
                relative = Path(source)
            doc.metadata["source"] = str(relative)
            documents.append(doc)

    if not documents:
        raise ValueError(f"No supported documents found under {docs_root}")

    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
    return splitter.split_documents(documents)


def build_vector_store(chunks: List[Document], vector_path: Path) -> None:
    from langchain_chroma import Chroma

    embedding_model = get_embeddings(
        model_name=os.getenv("EMBEDDING_MODEL"),
        device=os.getenv("EMBEDDING_DEVICE"),
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(vector_path),
        collection_name=COLLECTION_NAME,
    )


def main(argv: Iterable[str]) -> int:
    load_dotenv()
    args = parse_args(argv)
    docs_path, vector_path = resolve_paths(args)

    if args.rebuild and vector_path.exists():
        shutil.rmtree(vector_path)

    documents = load_documents(docs_path)
    chunks = chunk_documents(documents)
    build_vector_store(chunks, vector_path)

    print(f"Indexed {len(documents)} documents into {len(chunks)} chunks.")
    print(f"Vector store available at: {vector_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))