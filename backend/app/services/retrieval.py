import asyncio
import random
from collections import OrderedDict
import re
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import timing
from ..config import get_settings
from ..models import Document, DocumentChunk, DocumentFigure
from . import conversation, library, providers
from .figures import FIGURE_CHUNK_BASE
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
    # Set when this chunk describes a figure: {"id", "title", "width", "height"}.
    figure: dict[str, Any] | None = None

    def citation(self, index: int | None = None) -> dict[str, Any]:
        snippet = " ".join(self.text.split())
        extra = {"figure": self.figure} if self.figure else {}
        return {
            "index": index,
            "chunk_id": self.chunk_id,
            "document_id": str(self.document_id),
            "filename": self.filename,
            "page": self.page,
            "snippet": snippet if len(snippet) <= 280 else snippet[:277] + "...",
            "score": round(self.score, 3),
            "matched_by": self.matched_by,
            **extra,
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
        kind = ' kind="figure"' if e.figure else ""
        blocks.append(f'<source id="{i}" file="{name}"{page}{kind}>\n{_defuse(e.text)}\n</source>')
    return "\n\n".join(blocks)


async def _load_chunks(
    session: AsyncSession, user_id: uuid.UUID, document_ids: list[uuid.UUID], chunk_ids: list[str]
) -> dict[str, tuple[DocumentChunk, str]]:
    # Re-check access and section scope in the database; the vector store filter is not the only guard.
    rows = await session.execute(
        select(DocumentChunk, Document.filename)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.id.in_(chunk_ids),
            library.readable_by(user_id),
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


# Question embeddings started while the request saves the question (see prefetch_embedding),
# by question text. The answer takes its entry moments later; the size cap drops strays.
_prefetched: OrderedDict[str, asyncio.Future] = OrderedDict()
_PREFETCH_MAX = 32


def prefetch_embedding(question: str) -> None:
    """Start embedding a question now. Follow-ups that get rewritten are embedded later instead,
    since the rewritten text is what is searched."""
    if conversation.needs_rewrite(question) or question in _prefetched:
        return
    future = asyncio.ensure_future(asyncio.to_thread(providers.embedder().embed_query, question))
    future.add_done_callback(lambda f: f.cancelled() or f.exception())  # never "exception not retrieved"
    _prefetched[question] = future
    while len(_prefetched) > _PREFETCH_MAX:
        _prefetched.popitem(last=False)


async def _embed_query(query: str) -> list[float]:
    if (future := _prefetched.pop(query, None)) is not None:
        try:
            return await future
        except Exception:
            pass  # embed it again below
    return await asyncio.to_thread(providers.embedder().embed_query, query)


async def _vector_search(
    user_id: uuid.UUID, document_ids: list[uuid.UUID], query: str, top_k: int, min_score: float
) -> list[Any]:
    with timing.step("search: embed question"):
        vector = await _embed_query(query)
    with timing.step("search: pinecone"):
        matches = await asyncio.to_thread(
            providers.vector_store().query,
            providers.namespace_for(),
            vector,
            top_k,
            library.search_owners(user_id),
            [str(d) for d in document_ids],
        )
    return sorted((m for m in matches if m.score >= min_score), key=lambda m: m.score, reverse=True)


async def _timed(name: str, work):
    with timing.step(name):
        return await work


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
            _timed("search: keyword", keyword_search(session, user_id=user_id, document_ids=document_ids, query=query, top_k=candidates)),
        )
    else:
        dense, sparse = await _vector_search(user_id, document_ids, query, candidates, min_score), []

    dense_scores = {m.id: m.score for m in dense}
    keyword_ids = {chunk_id for chunk_id, _ in sparse}
    fused = reciprocal_rank_fusion([[m.id for m in dense], [chunk_id for chunk_id, _ in sparse]], settings.rrf_k)
    ranked = sorted(fused, key=fused.__getitem__, reverse=True)[:top_k]
    if not ranked:
        return []

    with timing.step("search: load passages"):
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
    await attach_figures(session, evidence)
    return evidence


async def attach_figures(session: AsyncSession, evidence: list[Evidence]) -> None:
    """Mark the evidence that describes a figure, so the answer can show the image."""
    ids = [e.chunk_id for e in evidence if "-fig-" in e.chunk_id]
    if not ids:
        return
    figures = {f.chunk_id: f for f in await session.scalars(select(DocumentFigure).where(DocumentFigure.chunk_id.in_(ids)))}
    for e in evidence:
        if (f := figures.get(e.chunk_id)) is not None:
            e.figure = {"id": str(f.id), "title": f.title, "width": f.width, "height": f.height}


def evenly_spaced(items: list, count: int) -> list:
    """`count` items spread evenly from first to last, in order (all of them if there are fewer)."""
    if count >= len(items):
        return list(items)
    if count <= 0:
        return []
    if count == 1:
        return [items[0]]
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


async def spread_chunks(
    session: AsyncSession, *, user_id: uuid.UUID, document_ids: list[uuid.UUID], limit: int
) -> list[Evidence]:
    """Text chunks spread evenly through each document, in reading order.

    For questions about a whole chapter ("explain this chapter", "what is this file about?"):
    such a question matches no single passage, and the best search matches all come from one
    or two places, while the chapter's definitions, theorems and examples run from its first
    page to its last.
    """
    if not document_ids or limit <= 0:
        return []
    rows = (
        await session.execute(
            select(DocumentChunk.id, DocumentChunk.document_id)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                library.readable_by(user_id),
                Document.id.in_(document_ids),
                DocumentChunk.chunk_index < FIGURE_CHUNK_BASE,  # the text, not figure descriptions
            )
            .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
        )
    ).all()
    by_document: dict[uuid.UUID, list[str]] = {}
    for chunk_id, document_id in rows:
        by_document.setdefault(document_id, []).append(chunk_id)
    if not by_document:
        return []
    # An even share per document; a short document's unused share goes to the others.
    chosen: list[str] = []
    remaining, documents_left = limit, len(by_document)
    for ids in sorted(by_document.values(), key=len):
        share = remaining // documents_left
        picked = evenly_spaced(ids, share)
        chosen.extend(picked)
        remaining -= len(picked)
        documents_left -= 1
    chunks = await _load_chunks(session, user_id, document_ids, chosen)
    order = {document_id: n for n, document_id in enumerate(document_ids)}
    ordered = sorted(chunks.values(), key=lambda c: (order.get(c[0].document_id, 0), c[0].chunk_index))
    return [Evidence(c.id, c.document_id, name, c.page, c.text, 0.0, "overview") for c, name in ordered]


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
            .where(
                library.readable_by(user_id),
                Document.id.in_(document_ids),
                DocumentChunk.chunk_index < FIGURE_CHUNK_BASE,  # quizzes come from the text
            )
        )
    )
    chosen = random.sample(ids, min(count, len(ids)))
    chunks = await _load_chunks(session, user_id, document_ids, chosen)
    ordered = sorted(chunks.values(), key=lambda c: (str(c[0].document_id), c[0].chunk_index))
    return [Evidence(c.id, c.document_id, name, c.page, c.text, 1.0, "sample") for c, name in ordered]
