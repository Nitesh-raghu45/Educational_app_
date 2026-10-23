import asyncio
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import storage
from ..config import get_settings
from ..errors import AppError
from ..models import Document, DocumentChunk, SectionDocument
from . import providers, usage
from .sections import get_owned_section

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
) -> Document:
    filename = Path(upload.filename or "upload").name[:255]
    extension = Path(filename).suffix.lower()
    if extension not in CONTENT_TYPES:
        raise AppError(415, "UNSUPPORTED_FILE_TYPE", "Supported files: PDF, DOCX, TXT, MD, PNG, JPG.")
    if section_id is not None:
        await get_owned_section(session, section_id, user_id)

    await usage.reserve(session, user_id, "upload")
    document_id = uuid.uuid4()
    path = storage.path_for(user_id, document_id, extension)
    size = await storage.save_upload(upload, path, get_settings().max_upload_mb * 1024 * 1024)
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
    if section_id is not None:
        await session.flush()
        session.add(SectionDocument(section_id=section_id, document_id=document_id))
    await session.commit()
    return doc


async def list_documents(session: AsyncSession, user_id: uuid.UUID) -> list[Document]:
    rows = await session.scalars(
        select(Document).where(Document.user_id == user_id).order_by(Document.created_at.desc())
    )
    return list(rows)


async def section_documents(session: AsyncSession, section_id: uuid.UUID, user_id: uuid.UUID) -> list[Document]:
    await get_owned_section(session, section_id, user_id)
    rows = await session.scalars(
        select(Document)
        .join(SectionDocument, SectionDocument.document_id == Document.id)
        .where(SectionDocument.section_id == section_id)
        .order_by(SectionDocument.created_at)
    )
    return list(rows)


async def attach(session: AsyncSession, section_id: uuid.UUID, document_id: uuid.UUID, user_id: uuid.UUID) -> Document:
    await get_owned_section(session, section_id, user_id)
    doc = await get_owned_document(session, document_id, user_id)
    if await session.get(SectionDocument, (section_id, document_id)) is None:
        session.add(SectionDocument(section_id=section_id, document_id=document_id))
        await session.commit()
    return doc


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
    chunk_ids = list(await session.scalars(select(DocumentChunk.id).where(DocumentChunk.document_id == doc.id)))
    if chunk_ids:
        await asyncio.to_thread(providers.vector_store().delete, providers.namespace_for(user_id), chunk_ids)
    await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))
    await session.execute(delete(SectionDocument).where(SectionDocument.document_id == doc.id))
    await session.execute(delete(Document).where(Document.id == doc.id))
    await session.commit()
    storage.delete(doc.storage_path)
