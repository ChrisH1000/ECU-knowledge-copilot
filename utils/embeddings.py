"""Embedding utilities shared across the application."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
NOMIC_PREFIX = {
    "document": "search_document: ",
    "query": "search_query: ",
}


class InstructionAwareEmbeddings(HuggingFaceEmbeddings):
    """Embeddings that optionally prepend task-specific prefixes."""

    def __init__(
        self,
        *,
        document_prefix: Optional[str] = None,
        query_prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix

    def embed_documents(self, texts):  # type: ignore[override]
        if self._document_prefix:
            texts = [f"{self._document_prefix}{text}" for text in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):  # type: ignore[override]
        if self._query_prefix:
            text = f"{self._query_prefix}{text}"
        return super().embed_query(text)


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
    model_kwargs = {"trust_remote_code": True}
    if device:
        model_kwargs["device"] = device

    encode_kwargs = {"normalize_embeddings": normalize}

    document_prefix: Optional[str] = None
    query_prefix: Optional[str] = None
    lowered_name = model_name.lower()
    if "nomic" in lowered_name:
        document_prefix = NOMIC_PREFIX["document"]
        query_prefix = NOMIC_PREFIX["query"]

    return InstructionAwareEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        document_prefix=document_prefix,
        query_prefix=query_prefix,
    )