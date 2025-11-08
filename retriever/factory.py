"""Factory helpers for configuring retrievers.

Provides a consistent interface for loading Chroma vector stores and
creating Maximal Marginal Relevance (MMR) retrievers per the PRD.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever


@dataclass(frozen=True)
class RetrieverConfig:
    """Declarative configuration for the MMR retriever.

    PRD specifies k=3 for top-k retrieval with MMR to balance relevance
    and diversity. fetch_k=9 pulls more candidates before MMR reranking.

    Attributes:
        k: Number of final chunks to return (default 3)
        fetch_k: Number of candidates to fetch before MMR filtering (default 9)
        lambda_mult: MMR diversity parameter (0=max diversity, 1=max relevance)
        collection_name: Chroma collection identifier
    """

    k: int = 3
    fetch_k: int = 9
    lambda_mult: float = 0.5
    collection_name: str = "ecu-knowledge"


def load_vectorstore(
    persist_directory: Path,
    embedding_function,
    config: Optional[RetrieverConfig] = None,
) -> Tuple[Chroma, VectorStoreRetriever]:
    """Instantiate a Chroma vector store and matching retriever.

    Loads a persisted Chroma index and creates an MMR retriever with
    the specified configuration (k=3, fetch_k=9 by default).

    Args:
        persist_directory: Path to the Chroma storage directory
        embedding_function: Embedding model instance (must match ingest config)
        config: Optional retriever configuration (uses defaults if None)

    Returns:
        tuple: (Chroma vectorstore, MMR retriever)
    """
    cfg = config or RetrieverConfig()
    path = Path(persist_directory)
    # Ensure directory exists (should already from ingest.py)
    path.mkdir(parents=True, exist_ok=True)

    # Load persisted Chroma collection
    vectorstore = Chroma(
        persist_directory=str(path),
        embedding_function=embedding_function,
        collection_name=cfg.collection_name,
    )

    # Create MMR retriever with PRD-specified parameters
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance for diversity
        search_kwargs={
            "k": cfg.k,              # Return top 3 chunks
            "fetch_k": cfg.fetch_k,  # Fetch 9 candidates for reranking
            "lambda_mult": cfg.lambda_mult,  # Balance relevance/diversity
        },
    )
    return vectorstore, retriever