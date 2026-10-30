import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import storage
from ..config import get_settings
from ..errors import AppError
from ..models import Document, DocumentChunk, DocumentHash, FigureScan, SectionDocument
from . import figures, providers, usage
from .sections import get_owned_section

logger = logging.getLogger(__name__)

CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


async def get_owned_document(session: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID) -> Document:
    doc = await session.scalar(select(Document).where(Document.id == document_id, Document.user_id == user_id))
    if doc is None:
        raise AppError(404, "DOCUMENT_NOT_FOUND", "This document does not exist.")
    return doc


async def save_upload(
    session: AsyncSession, user_id: uuid.UUID, upload: UploadFile, section_id: uuid.UUID | None
) -> "UploadResult":
    """Store an upload, or reuse the student's existing document when it is the same file."""
    filename = Path(upload.filename or "upload").name[:255]
    extension = Path(filename).suffix.lower()
    if extension not in CONTENT_TYPES:
        raise AppError(415, "UNSUPPORTED_FILE_TYPE", "Supported files: PDF, DOCX, TXT, MD, PNG, JPG.")
    if section_id is not None:
        await get_owned_section(session, section_id, user_id)

    await usage.reserve(session, user_id, "upload")
    document_id = uuid.uuid4()
    path = storage.path_for(user_id, document_id, extension)
    size, sha256 = await storage.save_upload(upload, path, get_settings().max_upload_mb * 1024 * 1024)

    existing = await _find_same_file(session, user_id, sha256, filename, size)
    if existing is not None:
        await usage.refund(session, user_id, "upload")  # nothing new to process
        return await _reuse(session, existing, path, section_id, user_id)

    doc = Document(
        id=document_id,
        user_id=user_id,
        filename=filename,
        content_type=CONTENT_TYPES[extension],
        size_bytes=size,
        storage_path=str(path),
        status="pending",
    )
    session.add(doc)
    await session.flush()
    session.add(DocumentHash(document_id=document_id, user_id=user_id, sha256=sha256))
    if section_id is not None:
        session.add(SectionDocument(section_id=section_id, document_id=document_id))
    await session.commit()
    return UploadResult(doc)


@dataclass
class UploadResult:
    doc: Document
    # None: a new document, to be processed. "existing": the same file was already in the
    # student's library and was attached instead. "already_in_chat": it was already attached.
    duplicate: str | None = None
    # Re-uploading restored a file whose figures were never extracted: extract them now.
    extract_figures: bool = False


# Documents a re-upload can reuse; a failed one is processed again from the new upload.
_REUSABLE = ("pending", "processing", "ready")


async def _find_same_file(
    session: AsyncSession, user_id: uuid.UUID, sha256: str, filename: str, size: int
) -> Document | None:
    doc = await session.scalar(
        select(Document)
        .join(DocumentHash, DocumentHash.document_id == Document.id)
        .where(DocumentHash.user_id == user_id, DocumentHash.sha256 == sha256, Document.status.in_(_REUSABLE))
        .order_by(Document.created_at)
        .limit(1)
    )
    if doc is not None:
        return doc
    # Uploaded before fingerprints were kept, with its file since lost (hosts like Render wipe
    # uploads on restart): the same name and exact byte size is taken to be the same file.
    doc = await session.scalar(
        select(Document)
        .outerjoin(DocumentHash, DocumentHash.document_id == Document.id)
        .where(
            Document.user_id == user_id,
            Document.filename == filename,
            Document.size_bytes == size,
            Document.status.in_(_REUSABLE),
            DocumentHash.document_id.is_(None),
        )
        .order_by(Document.created_at)
        .limit(1)
    )
    if doc is not None:
        session.add(DocumentHash(document_id=doc.id, user_id=user_id, sha256=sha256))
    return doc


async def _reuse(
    session: AsyncSession, doc: Document, new_path: Path, section_id: uuid.UUID | None, user_id: uuid.UUID
) -> UploadResult:
    """Attach the student's existing copy of an uploaded file instead of processing it again."""
    already = section_id is not None and await session.get(SectionDocument, (section_id, doc.id)) is not None
    if section_id is not None and not already:
        session.add(SectionDocument(section_id=section_id, document_id=doc.id))
    if Path(doc.storage_path).exists():
        storage.delete(str(new_path))
    else:
        # The server lost its copy (e.g. a restart on Render): keep this one, so figures and
        # re-embedding can use it again.
        doc.storage_path = str(new_path)
    # Figures never extracted (uploaded before figure extraction existed) or extraction failed:
    # now that the file is here again, extract them.
    scan = await session.get(FigureScan, doc.id)
    extract = doc.status == "ready" and get_settings().figures_enabled and (scan is None or scan.status == "failed")
    if extract:
        await figures.mark_processing(session, doc.id)
    await session.commit()
    await session.refresh(doc)
    doc = (await with_figure_counts(session, [doc]))[0]
    return UploadResult(doc, "already_in_chat" if already else "existing", extract)


async def backfill_hashes() -> None:
    """Fingerprint documents uploaded before fingerprints were kept, if their file is still here."""
    from ..db import SessionLocal

    try:
        async with SessionLocal() as session:
            rows = (
                await session.execute(
                    select(Document.id, Document.user_id, Document.storage_path)
                    .outerjoin(DocumentHash, DocumentHash.document_id == Document.id)
                    .where(DocumentHash.document_id.is_(None))
                )
            ).all()
            added = 0
            for doc_id, user_id, path in rows:
                if sha256 := await asyncio.to_thread(storage.file_sha256, path):
                    session.add(DocumentHash(document_id=doc_id, user_id=user_id, sha256=sha256))
                    added += 1
            await session.commit()
        if added:
            logger.info("Fingerprinted %d earlier upload(s) for duplicate detection", added)
    except Exception:
        logger.warning("Could not fingerprint earlier uploads", exc_info=True)


async def with_figure_counts(session: AsyncSession, docs: list[Document]) -> list[Document]:
    """Adds `figure_status` and `figure_count` (read by DocumentOut) to each document."""
    counts = await figures.figure_counts(session, [d.id for d in docs])
    for doc in docs:
        doc.figure_status, doc.figure_count = counts.get(doc.id, (None, 0))
    return docs


async def list_documents(session: AsyncSession, user_id: uuid.UUID) -> list[Document]:
    rows = await session.scalars(
        select(Document).where(Document.user_id == user_id).order_by(Document.created_at.desc())
    )
    return await with_figure_counts(session, list(rows))


async def section_documents(session: AsyncSession, section_id: uuid.UUID, user_id: uuid.UUID) -> list[Document]:
    await get_owned_section(session, section_id, user_id)
    rows = await session.scalars(
        select(Document)
        .join(SectionDocument, SectionDocument.document_id == Document.id)
        .where(SectionDocument.section_id == section_id)
        .order_by(SectionDocument.created_at)
    )
    return await with_figure_counts(session, list(rows))


async def attach(session: AsyncSession, section_id: uuid.UUID, document_id: uuid.UUID, user_id: uuid.UUID) -> Document:
    await get_owned_section(session, section_id, user_id)
    doc = await get_owned_document(session, document_id, user_id)
    if await session.get(SectionDocument, (section_id, document_id)) is None:
        session.add(SectionDocument(section_id=section_id, document_id=document_id))
        await session.commit()
    return (await with_figure_counts(session, [doc]))[0]


async def detach(session: AsyncSession, section_id: uuid.UUID, document_id: uuid.UUID, user_id: uuid.UUID) -> None:
    await get_owned_section(session, section_id, user_id)
    await session.execute(
        delete(SectionDocument).where(
            SectionDocument.section_id == section_id, SectionDocument.document_id == document_id
        )
    )
    await session.commit()


async def delete_document(session: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID) -> None:
    doc = await get_owned_document(session, document_id, user_id)
    await figures.clear_figures(session, doc)
    chunk_ids = list(await session.scalars(select(DocumentChunk.id).where(DocumentChunk.document_id == doc.id)))
    if chunk_ids:
        await asyncio.to_thread(providers.vector_store().delete, providers.namespace_for(user_id), chunk_ids)
    await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))
    await session.execute(delete(SectionDocument).where(SectionDocument.document_id == doc.id))
    await session.execute(delete(FigureScan).where(FigureScan.document_id == doc.id))
    await session.execute(delete(DocumentHash).where(DocumentHash.document_id == doc.id))
    await session.execute(delete(Document).where(Document.id == doc.id))
    await session.commit()
    storage.delete(doc.storage_path)
