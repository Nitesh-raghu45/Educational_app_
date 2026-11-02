"""A copy of each uploaded file in the database, so a host restart that wipes the disk (like
Render's free plan) can't lose it: processing, figure extraction and re-embedding restore the
file from here when it's missing on disk.

Files are split into parts of a few MB, each written in its own INSERT, so no single statement
comes near MySQL's max_allowed_packet.
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Document, DocumentFilePart

logger = logging.getLogger(__name__)

PART_BYTES = 4 * 1024 * 1024


def _read_parts(path: Path) -> list[bytes]:
    with path.open("rb") as f:
        return list(iter(lambda: f.read(PART_BYTES), b""))


async def save(session: AsyncSession, document_id: uuid.UUID, path: Path) -> None:
    """Store the file's copy (committed by the caller)."""
    for n, data in enumerate(await asyncio.to_thread(_read_parts, path)):
        session.add(DocumentFilePart(document_id=document_id, part=n, data=data))
        await session.flush()  # one part per statement


async def has_copy(session: AsyncSession, document_id: uuid.UUID) -> bool:
    found = await session.scalar(
        select(DocumentFilePart.part).where(DocumentFilePart.document_id == document_id).limit(1)
    )
    return found is not None


def _write(path: Path, parts: list[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as out:
        for data in parts:
            out.write(data)
    os.replace(partial, path)  # never leaves a half-written file at the real path


async def ensure_local(session: AsyncSession, doc: Document) -> bool:
    """True if the document's file is on disk, restoring it from the database if needed."""
    path = Path(doc.storage_path)
    if path.exists():
        return True
    numbers = list(
        await session.scalars(
            select(DocumentFilePart.part).where(DocumentFilePart.document_id == doc.id).order_by(DocumentFilePart.part)
        )
    )
    if not numbers:
        return False  # uploaded before copies were kept
    parts = []
    for n in numbers:
        row = await session.get(DocumentFilePart, (doc.id, n))
        parts.append(row.data)
        session.expunge(row)
    await asyncio.to_thread(_write, path, parts)
    logger.info("Restored %r (document %s) from the database", doc.filename, doc.id)
    return True


async def remove(session: AsyncSession, document_id: uuid.UUID) -> None:
    await session.execute(delete(DocumentFilePart).where(DocumentFilePart.document_id == document_id))
