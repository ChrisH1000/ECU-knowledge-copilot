"""Embedding utilities shared across the application.

Provides a factory for HuggingFace embeddings with optional instruction
prefixes. Supports the Nomic embedding models that require task-specific
prefixes ("search_document:" and "search_query:") for optimal performance.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings


# PRD default: bge-small-en-v1.5 for cost-aware embeddings
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Nomic models require these prefixes for document vs. query embeddings
NOMIC_PREFIX = {
    "document": "search_document: ",
    "query": "search_query: ",
}


class InstructionAwareEmbeddings(HuggingFaceEmbeddings):
    """Embeddings that optionally prepend task-specific prefixes.

    Some embedding models (notably Nomic) require instruction prefixes
    to distinguish between document and query contexts. This wrapper
    automatically prepends the appropriate prefix before encoding.

    Attributes:
        _document_prefix: Prefix for document/chunk embeddings (e.g., "search_document: ")
        _query_prefix: Prefix for query embeddings (e.g., "search_query: ")
    """

    def __init__(
        self,
        *,
        document_prefix: Optional[str] = None,
        query_prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize with optional instruction prefixes.

        Args:
            document_prefix: Text to prepend when embedding documents
            query_prefix: Text to prepend when embedding queries
            **kwargs: Additional arguments passed to HuggingFaceEmbeddings
        """
        super().__init__(**kwargs)
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix

    def embed_documents(self, texts):  # type: ignore[override]
        """Embed a list of document texts with optional prefix.

        Args:
            texts: List of document strings to embed

        Returns:
            List of embedding vectors
        """
        # Prepend document instruction if configured
        if self._document_prefix:
            texts = [f"{self._document_prefix}{text}" for text in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):  # type: ignore[override]
        """Embed a single query text with optional prefix.

        Args:
            text: Query string to embed

        Returns:
            Embedding vector
        """
        # Prepend query instruction if configured
        if self._query_prefix:
            text = f"{self._query_prefix}{text}"
        return super().embed_query(text)


def get_embeddings(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    normalize: bool = True,
) -> HuggingFaceEmbeddings:
    """Return a configured embedding model instance.

    Main entry point for creating embeddings. Delegates to cached factory
    to avoid reloading model weights on repeated calls.

    Args:
        model_name: HuggingFace model ID (defaults to bge-small-en-v1.5)
        device: Compute device ("cpu", "cuda", etc.) or None for auto
        normalize: Whether to L2-normalize embeddings (default True)

    Returns:
        HuggingFaceEmbeddings: Configured embedding model (possibly with prefixes)
    """
    return _embedding_factory(model_name or DEFAULT_EMBEDDING_MODEL, device, normalize)


@lru_cache(maxsize=8)
def _embedding_factory(
    model_name: str,
    device: Optional[str],
    normalize: bool,
) -> HuggingFaceEmbeddings:
    """Cached factory for embedding model instances.

    Builds the embedding model with appropriate configuration, including
    Nomic-specific instruction prefixes when detected.

    Args:
        model_name: HuggingFace model ID
        device: Target compute device (or None for auto)
        normalize: Whether to normalize embeddings

    Returns:
        InstructionAwareEmbeddings: Configured model ready for use
    """
    # Enable trust_remote_code for models like Nomic that use custom code
    model_kwargs = {"trust_remote_code": True}
    if device:
        model_kwargs["device"] = device

    # Configure encoding behavior (normalization for cosine similarity)
    encode_kwargs = {"normalize_embeddings": normalize}

    # Detect Nomic models and add instruction prefixes
    document_prefix: Optional[str] = None
    query_prefix: Optional[str] = None
    lowered_name = model_name.lower()
    if "nomic" in lowered_name:
        document_prefix = NOMIC_PREFIX["document"]
        query_prefix = NOMIC_PREFIX["query"]

    # Return instruction-aware wrapper with all configuration
    return InstructionAwareEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        document_prefix=document_prefix,
        query_prefix=query_prefix,
    )