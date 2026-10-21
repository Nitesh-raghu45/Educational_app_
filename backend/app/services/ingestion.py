"""Document ingestion: parse -> chunk -> embed (MiniLM, local) -> upsert to Pinecone -> mark ready."""

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, select

from ..config import get_settings
from ..db import SessionLocal
from ..models import Document, DocumentChunk, utcnow
from . import providers

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """A problem with the file itself; the message is shown to the student."""


@dataclass
class Chunk:
    page: int | None
    text: str


def extract_text(path: Path) -> list[tuple[int | None, str]]:
    """Returns (page number or None, text) pairs."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
    if ext == ".docx":
        import docx

        doc = docx.Document(str(path))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return [(None, "\n".join(parts))]
    if ext in (".txt", ".md"):
        return [(None, path.read_text(encoding="utf-8", errors="replace"))]
    if ext in (".png", ".jpg", ".jpeg"):
        try:
            import pytesseract
            from PIL import Image

            return [(None, pytesseract.image_to_string(Image.open(path)))]
        except pytesseract.TesseractNotFoundError as exc:
            raise IngestionError("Image text recognition (Tesseract OCR) is not installed on the server.") from exc
    raise IngestionError(f"Unsupported file type: {ext}")


# Coarsest to finest: paragraphs, lines, sentences, clauses, words, and characters as a last resort.
SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


def _split_pieces(text: str, size: int, separators: tuple[str, ...]) -> list[str]:
    """Cut text into pieces of at most `size` characters, using the coarsest separator that works.

    Only pieces that are still too long are split again with the next, finer separator.
    """
    if len(text) <= size:
        return [text]
    separator, finer = separators[0], separators[1:]
    if not separator:
        return [text[i : i + size] for i in range(0, len(text), size)]
    if separator not in text:
        return _split_pieces(text, size, finer)
    parts = text.split(separator)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        # Keep the separator with the text before it, so joined pieces read like the original.
        piece = part + separator if i < len(parts) - 1 else part
        if piece:
            pieces.extend([piece] if len(piece) <= size else _split_pieces(piece, size, finer))
    return pieces


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Recursive chunking: keep paragraphs (then lines, sentences, …) whole whenever they fit,
    merging neighbours up to `size` characters. Consecutive chunks share up to `overlap`
    characters of whole trailing sentences/lines, so context isn't lost at a boundary."""
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    window: list[str] = []
    length = 0
    for piece in _split_pieces(text, size, SEPARATORS):
        if window and length + len(piece) > size:
            chunks.append("".join(window).strip())
            # Carry whole trailing pieces into the next chunk as overlap.
            while window and (length > overlap or length + len(piece) > size):
                length -= len(window.pop(0))
        window.append(piece)
        length += len(piece)
    if window:
        chunks.append("".join(window).strip())
    return [chunk for chunk in chunks if chunk]


def chunk_pages(pages: list[tuple[int | None, str]], size: int, overlap: int) -> list[Chunk]:
    return [Chunk(page, piece) for page, text in pages for piece in split_text(text, size, overlap)]


async def ingest_document(document_id: uuid.UUID) -> None:
    settings = get_settings()
    async with SessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            return
        doc.status, doc.error, doc.updated_at = "processing", None, utcnow()
        await session.commit()
        namespace = providers.namespace_for(doc.user_id)
        try:
            if not Path(doc.storage_path).exists():
                raise IngestionError("The uploaded file is no longer on the server (it restarted). Please upload it again.")
            pages = await asyncio.to_thread(extract_text, Path(doc.storage_path))
            chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
            if not chunks:
                raise IngestionError(
                    "No text could be extracted. Scanned PDFs aren't supported yet; upload page images instead."
                )

            # Re-ingestion: drop previous vectors and rows first.
            old_ids = list(await session.scalars(select(DocumentChunk.id).where(DocumentChunk.document_id == doc.id)))
            if old_ids:
                await asyncio.to_thread(providers.vector_store().delete, namespace, old_ids)
                await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))

            vectors = await asyncio.to_thread(providers.embedder().embed_documents, [c.text for c in chunks])
            rows = [
                DocumentChunk(id=f"{doc.id.hex}-{i}", document_id=doc.id, chunk_index=i, page=c.page, text=c.text)
                for i, c in enumerate(chunks)
            ]
            items = [
                (
                    row.id,
                    vector,
                    {"document_id": str(doc.id), "chunk_index": row.chunk_index, "page": row.page or 0},
                )
                for row, vector in zip(rows, vectors)
            ]
            await asyncio.to_thread(providers.vector_store().upsert, namespace, items)
            session.add_all(rows)
            doc.status, doc.chunk_count, doc.updated_at = "ready", len(rows), utcnow()
            await session.commit()
            logger.info("Ingested document %s (%d chunks)", doc.id, len(rows))
        except Exception as exc:
            logger.exception("Ingestion failed for document %s", document_id)
            await session.rollback()
            doc = await session.get(Document, document_id, populate_existing=True)
            if doc is not None:
                message = str(exc) if isinstance(exc, IngestionError) else "Processing failed. Try uploading again."
                doc.status, doc.error, doc.updated_at = "failed", message, utcnow()
                await session.commit()
