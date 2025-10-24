"""Factory helpers for configuring retrievers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever


@dataclass(frozen=True)
class RetrieverConfig:
    """Declarative configuration for the MMR retriever."""

    k: int = 3
    fetch_k: int = 9
    lambda_mult: float = 0.5
    collection_name: str = "ecu-knowledge"


def load_vectorstore(
    persist_directory: Path,
    embedding_function,
    config: Optional[RetrieverConfig] = None,
) -> Tuple[Chroma, VectorStoreRetriever]:
    """Instantiate a Chroma vector store and matching retriever."""

    cfg = config or RetrieverConfig()
    path = Path(persist_directory)
    path.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        persist_directory=str(path),
        embedding_function=embedding_function,
        collection_name=cfg.collection_name,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": cfg.k,
            "fetch_k": cfg.fetch_k,
            "lambda_mult": cfg.lambda_mult,
        },
    )
    return vectorstore, retriever