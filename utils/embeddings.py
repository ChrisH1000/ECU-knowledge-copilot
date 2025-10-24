"""Embedding utilities shared across the application."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embeddings(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    normalize: bool = True,
) -> HuggingFaceEmbeddings:
    """Return a configured embedding model instance."""

    return _embedding_factory(model_name or DEFAULT_EMBEDDING_MODEL, device, normalize)


@lru_cache(maxsize=8)
def _embedding_factory(
    model_name: str,
    device: Optional[str],
    normalize: bool,
) -> HuggingFaceEmbeddings:
    model_kwargs = {"device": device} if device else {}
    encode_kwargs = {"normalize_embeddings": normalize}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )