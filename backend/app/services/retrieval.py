import asyncio
import random
import re
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..models import Document, DocumentChunk
from . import providers
from .keyword import keyword_search


@dataclass
class Evidence:
    chunk_id: str
    document_id: uuid.UUID
    filename: str
    page: int | None
    text: str
    # Vector similarity (0 when the chunk was found only by keyword search).
    score: float
    # "vector", "keyword" or "both": which search found this chunk.
    matched_by: str = "vector"

    def citation(self, index: int | None = None) -> dict[str, Any]:
        snippet = " ".join(self.text.split())
        return {
            "index": index,
            "chunk_id": self.chunk_id,
            "document_id": str(self.document_id),
            "filename": self.filename,
            "page": self.page,
            "snippet": snippet if len(snippet) <= 280 else snippet[:277] + "...",
            "score": round(self.score, 3),
            "matched_by": self.matched_by,
        }


# Document text and file names are untrusted: they must not be able to close a <source> block
# early or fake a new one, which could pass their contents off as instructions.
_TAG_RE = re.compile(r"<\s*/?\s*source\b", re.IGNORECASE)


def _defuse(text: str) -> str:
    return _TAG_RE.sub(lambda m: m.group(0).replace("<", "‹"), text)


def format_sources(evidence: list[Evidence]) -> str:
    blocks = []
    for i, e in enumerate(evidence, 1):
        name = _defuse(e.filename).replace('"', "'").replace(">", "›")
        page = f' page="{e.page}"' if e.page else ""
        blocks.append(f'<source id="{i}" file="{name}"{page}>\n{_defuse(e.text)}\n</source>')
    return "\n\n".join(blocks)


async def _load_chunks(
    session: AsyncSession, user_id: uuid.UUID, document_ids: list[uuid.UUID], chunk_ids: list[str]
) -> dict[str, tuple[DocumentChunk, str]]:
    # Re-check ownership and section scope in the database; the vector store filter is not the only guard.
    rows = await session.execute(
        select(DocumentChunk, Document.filename)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.id.in_(chunk_ids),
            Document.user_id == user_id,
            Document.id.in_(document_ids),
        )
    )
    return {chunk.id: (chunk, filename) for chunk, filename in rows}


def reciprocal_rank_fusion(rankings: list[list[str]], k: int) -> dict[str, float]:
    """Merge ranked id lists: each list contributes 1 / (k + rank). Robust to different score scales."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, 1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return fused


async def _vector_search(
    user_id: uuid.UUID, document_ids: list[uuid.UUID], query: str, top_k: int, min_score: float
) -> list[Any]:
    vector = await asyncio.to_thread(providers.embedder().embed_query, query)
    matches = await asyncio.to_thread(
        providers.vector_store().query,
        providers.namespace_for(user_id),
        vector,
        top_k,
        [str(d) for d in document_ids],
    )
    return sorted((m for m in matches if m.score >= min_score), key=lambda m: m.score, reverse=True)


async def retrieve(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    document_ids: list[uuid.UUID],
    query: str,
    top_k: int,
    min_score: float,
) -> list[Evidence]:
    """Hybrid retrieval: Pinecone vector search + BM25 keyword search, merged with reciprocal rank fusion.

    Vector matches below `min_score` and keyword matches sharing too few query terms are dropped
    first, so an off-topic question still finds no evidence.
    """
    if not document_ids:
        return []
    settings = get_settings()
    candidates = max(top_k, settings.retrieval_candidates)
    if settings.hybrid_search:
        dense, sparse = await asyncio.gather(
            _vector_search(user_id, document_ids, query, candidates, min_score),
            keyword_search(session, user_id=user_id, document_ids=document_ids, query=query, top_k=candidates),
        )
    else:
        dense, sparse = await _vector_search(user_id, document_ids, query, candidates, min_score), []

    dense_scores = {m.id: m.score for m in dense}
    keyword_ids = {chunk_id for chunk_id, _ in sparse}
    fused = reciprocal_rank_fusion([[m.id for m in dense], [chunk_id for chunk_id, _ in sparse]], settings.rrf_k)
    ranked = sorted(fused, key=fused.__getitem__, reverse=True)[:top_k]
    if not ranked:
        return []

    chunks = await _load_chunks(session, user_id, document_ids, ranked)
    evidence = []
    for chunk_id in ranked:
        if chunk_id in chunks:
            chunk, filename = chunks[chunk_id]
            matched_by = "both" if chunk_id in dense_scores and chunk_id in keyword_ids else (
                "vector" if chunk_id in dense_scores else "keyword"
            )
            evidence.append(
                Evidence(
                    chunk.id, chunk.document_id, filename, chunk.page, chunk.text,
                    dense_scores.get(chunk_id, 0.0), matched_by,
                )
            )
    return evidence


async def leading_chunks(
    session: AsyncSession, *, user_id: uuid.UUID, document_ids: list[uuid.UUID], limit: int
) -> list[Evidence]:
    """The opening chunks of each document, spread evenly across documents.

    Similarity search can't answer questions about a document as a whole ("what is this file
    about?", "summarize it"): the question shares no words or meaning with any single passage.
    """
    if not document_ids or limit <= 0:
        return []
    per_document = max(1, limit // len(document_ids))
    rows = await session.execute(
        select(DocumentChunk, Document.filename)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            Document.user_id == user_id,
            Document.id.in_(document_ids),
            DocumentChunk.chunk_index < per_document,
        )
        .order_by(DocumentChunk.chunk_index, Document.created_at)
        .limit(limit)
    )
    return [
        Evidence(chunk.id, chunk.document_id, filename, chunk.page, chunk.text, 0.0, "overview")
        for chunk, filename in rows
    ]


async def sample_chunks(
    session: AsyncSession, *, user_id: uuid.UUID, document_ids: list[uuid.UUID], count: int
) -> list[Evidence]:
    """Random spread of chunks across the documents, for quizzes without a topic."""
    if not document_ids:
        return []
    ids = list(
        await session.scalars(
            select(DocumentChunk.id)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.user_id == user_id, Document.id.in_(document_ids))
        )
    )
    chosen = random.sample(ids, min(count, len(ids)))
    chunks = await _load_chunks(session, user_id, document_ids, chosen)
    ordered = sorted(chunks.values(), key=lambda c: (str(c[0].document_id), c[0].chunk_index))
    return [Evidence(c.id, c.document_id, name, c.page, c.text, 1.0, "sample") for c, name in ordered]
