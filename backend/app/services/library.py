"""The shared library: documents an admin uploads (e.g. NCERT textbooks) that every user can add to
their own chats and ask about. Everyone's own uploads stay private to them.

Access rule, used wherever documents are read: a user may read a document they own, or one in
the library (`readable_by`). Library documents are still owned by the admin who uploaded them;
their vectors are tagged with the owner LIBRARY_OWNER instead of a user id, so a search matches
"mine or the library's" with no extra database query (`search_owners`).
"""

import time
import uuid

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..errors import AppError
from ..models import Document, LibraryDocument, User

LIBRARY_OWNER = "library"


def readable_by(user_id: uuid.UUID):
    """SQL condition: documents this user may read (their own, or the shared library's)."""
    return or_(Document.user_id == user_id, Document.id.in_(select(LibraryDocument.document_id)))


def search_owners(user_id: uuid.UUID) -> list[str]:
    """Vector owners a user's searches may match."""
    return [str(user_id), LIBRARY_OWNER]


async def vector_owner(session: AsyncSession, doc: Document) -> str:
    """The owner tag for a document's vectors."""
    return LIBRARY_OWNER if await session.get(LibraryDocument, doc.id) is not None else str(doc.user_id)


async def get_readable_document(session: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID) -> Document:
    doc = await session.scalar(select(Document).where(Document.id == document_id, readable_by(user_id)))
    if doc is None:
        raise AppError(404, "DOCUMENT_NOT_FOUND", "This document does not exist.")
    return doc


def require_admin(user: User) -> User:
    if not user.is_admin:
        raise AppError(403, "ADMIN_ONLY", "Only an admin can change the shared library.")
    return user


# Admin user ids, looked up by email (ADMIN_EMAILS) and cached briefly: daily limits check this
# on every question, and a new admin account is picked up within a minute.
_ADMIN_CACHE_SECONDS = 60
_admin_cache: tuple[float, frozenset[str], frozenset[uuid.UUID]] | None = None


async def is_admin_id(session: AsyncSession, user_id: uuid.UUID) -> bool:
    global _admin_cache
    emails = get_settings().admin_email_set
    if not emails:
        return False
    now = time.monotonic()
    if _admin_cache is None or _admin_cache[1] != emails or now - _admin_cache[0] > _ADMIN_CACHE_SECONDS:
        ids = frozenset(await session.scalars(select(User.id).where(func.lower(User.email).in_(emails))))
        _admin_cache = (now, emails, ids)
    return user_id in _admin_cache[2]


async def library_flags(session: AsyncSession, document_ids: list[uuid.UUID]) -> dict[uuid.UUID, LibraryDocument]:
    if not document_ids:
        return {}
    rows = await session.scalars(select(LibraryDocument).where(LibraryDocument.document_id.in_(document_ids)))
    return {row.document_id: row for row in rows}


async def list_library(session: AsyncSession, *, include_unready: bool) -> list[Document]:
    """Library documents, with `grade` and `subject` set on each. Users see only ready ones."""
    query = select(Document).join(LibraryDocument, LibraryDocument.document_id == Document.id)
    if not include_unready:
        query = query.where(Document.status == "ready")
    return list(await session.scalars(query.order_by(LibraryDocument.grade, LibraryDocument.subject, Document.filename)))


async def share(session: AsyncSession, doc: Document, grade: str, subject: str) -> bool:
    """Put a document in the library (or update its class and subject). True when it was not in
    the library before: its vectors then still carry the uploader's id and must be re-indexed."""
    entry = await session.get(LibraryDocument, doc.id)
    if entry is None:
        session.add(LibraryDocument(document_id=doc.id, grade=grade, subject=subject))
    else:
        entry.grade, entry.subject = grade, subject
    await session.commit()
    return entry is None
