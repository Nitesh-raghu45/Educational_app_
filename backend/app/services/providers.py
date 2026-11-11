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
    def query(
        self, namespace: str, vector: list[float], top_k: int, owners: list[str], document_ids: list[str]
    ) -> list[Any]: ...
    def delete(self, namespace: str, ids: list[str]) -> None: ...


class LLM(Protocol):
    def stream(self, messages: list[dict[str, str]], model: str, **kwargs: Any) -> AsyncIterator[str]: ...
    async def complete(self, messages: list[dict[str, str]], model: str, **kwargs: Any) -> str: ...


class Vision(Protocol):
    async def describe(self, image_jpeg: bytes, prompt: str) -> str: ...
    async def transcribe(self, image_jpeg: bytes, prompt: str) -> str: ...


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
        from .vision import create_vision

        _vision = create_vision(get_settings())
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


# The embedding model used before model tracking existed. Documents embedded before then are
# counted as made with it (see main.reembed_outdated_documents).
LEGACY_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.split("/")[-1].lower()).strip("-")


def namespace_for(model: str | None = None) -> str:
    """All students share one Pinecone namespace per embedding model (vectors from different
    models can't be compared, so they must never be searched together). Each vector carries its
    owner (a user id, or "library" for the shared library) and every search filters on it, so
    students only find their own documents and the library's. One namespace per student would hit Pinecone's free-plan cap of 100 namespaces."""
    return shared_namespace(model_slug(model or get_settings().embedding_model))


def shared_namespace(slug: str) -> str:
    return f"docs--{slug}"


def vector_metadata(owner: str, document_id: Any, chunk_index: int, page: int | None) -> dict[str, Any]:
    """Metadata stored with every vector. `user_id` holds the owner (see library.vector_owner),
    which scopes searches."""
    return {"user_id": owner, "document_id": str(document_id), "chunk_index": chunk_index, "page": page or 0}
