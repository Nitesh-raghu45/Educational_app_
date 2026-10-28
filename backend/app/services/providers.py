"""Embedding model, vector store and LLM clients.

Created lazily so the API starts without API keys or the embedding model downloaded, and
replaceable with fakes in tests via `override()`.
"""

import re
from typing import Any, AsyncIterator, Protocol

from ..config import get_settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    def upsert(self, namespace: str, items: list[tuple[str, list[float], dict[str, Any]]]) -> None: ...
    def query(self, namespace: str, vector: list[float], top_k: int, document_ids: list[str]) -> list[Any]: ...
    def delete(self, namespace: str, ids: list[str]) -> None: ...


class LLM(Protocol):
    def stream(self, messages: list[dict[str, str]], model: str, **kwargs: Any) -> AsyncIterator[str]: ...
    async def complete(self, messages: list[dict[str, str]], model: str, **kwargs: Any) -> str: ...


class Vision(Protocol):
    async def describe(self, image_jpeg: bytes, prompt: str) -> str: ...


_embedder: Embedder | None = None
_store: VectorStore | None = None
_llm: LLM | None = None
_vision: Vision | None = None


def embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        from .embeddings import create_embedder

        settings = get_settings()
        _embedder = create_embedder(settings.embedding_model, settings.embedding_cache_dir, settings.embedding_batch_size)
    return _embedder


def vector_store() -> VectorStore:
    global _store
    if _store is None:
        from .vectorstore import PineconeStore

        _store = PineconeStore(get_settings())
    return _store


def llm() -> LLM:
    global _llm
    if _llm is None:
        from .llm import GroqLLM

        _llm = GroqLLM(get_settings())
    return _llm


def vision() -> Vision:
    global _vision
    if _vision is None:
        from .llm import GroqVision

        _vision = GroqVision(get_settings())
    return _vision


def override(
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    llm: LLM | None = None,
    vision: Vision | None = None,
) -> None:
    global _embedder, _store, _llm, _vision
    _embedder, _store, _llm, _vision = embedder, store, llm, vision


# Vectors made by this model live in the original namespace names, so they stay usable if the
# model is switched back.
LEGACY_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def namespace_for(user_id: Any, model: str | None = None) -> str:
    """Each student's vectors live in their own Pinecone namespace, one per embedding model:
    vectors from different models can't be compared, so they must never be searched together."""
    model = model or get_settings().embedding_model
    if model == LEGACY_EMBEDDING_MODEL:
        return f"user-{user_id}"
    slug = re.sub(r"[^a-z0-9]+", "-", model.split("/")[-1].lower()).strip("-")
    return f"user-{user_id}--{slug}"
