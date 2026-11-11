"""The shared library: everyone can list it; only admins (ADMIN_EMAILS) can add or remove books."""

import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..models import Document, User
from ..ratelimit import enforce_rate_limit
from ..schemas import DocumentOut
from ..security import current_user
from ..errors import AppError
from ..services import documents, library, stored_files
from ..services.jobs import enqueue_figures, enqueue_ingestion

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/library", tags=["library"])


def admin_user(user: User = Depends(current_user)) -> User:
    return library.require_admin(user)


@router.get("", response_model=list[DocumentOut])
async def list_library(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    """Every library book. Users see the ready ones; admins also see ones processing or failed."""
    docs = await library.list_library(session, include_unready=user.is_admin)
    return await documents.with_figure_counts(session, docs)


@router.post("", response_model=DocumentOut, status_code=201)
async def add_to_library(
    file: UploadFile = File(...),
    grade: str = Form("", max_length=20),
    subject: str = Form("", max_length=60),
    user: User = Depends(admin_user),
    session: AsyncSession = Depends(get_session),
) -> Document:
    """Upload a book (or chapter) to the shared library. Processed one at a time, like uploads."""
    await enforce_rate_limit(user.id, "library-uploads", 120)
    result = await documents.save_upload(session, user.id, file, None)
    doc = result.doc
    newly_shared = await library.share(session, doc, grade.strip(), subject.strip())
    if result.duplicate is None or newly_shared:
        # New, or the admin's own document joining the library: its vectors must carry the
        # library tag before other users can search it, so it's processed (again).
        await enqueue_ingestion(doc.id)
    elif result.extract_figures:
        enqueue_figures(doc.id)
    logger.info("Library upload %s (%r, class %r, %r)%s", doc.id, doc.filename, grade, subject,
                f" (same file as before: {result.duplicate})" if result.duplicate else "")
    doc = (await documents.with_figure_counts(session, [doc]))[0]
    doc.duplicate = None if newly_shared else result.duplicate
    return doc


@router.post("/{document_id}/reprocess", response_model=DocumentOut)
async def reprocess(
    document_id: uuid.UUID, user: User = Depends(admin_user), session: AsyncSession = Depends(get_session)
) -> Document:
    """Read a library book again: math pages (saved ones are reused) and figures. For books
    processed while the vision models were at their limits."""
    doc = await documents.get_owned_document(session, document_id, user.id)
    if not await stored_files.ensure_local(session, doc):
        raise AppError(409, "FILE_GONE", "This book's file is no longer stored. Delete it and upload it again.")
    # It stays "ready" (students keep using it) until processing starts.
    await enqueue_ingestion(doc.id)
    return (await documents.with_figure_counts(session, [doc]))[0]


@router.delete("/{document_id}", status_code=204)
async def remove_from_library(
    document_id: uuid.UUID, user: User = Depends(admin_user), session: AsyncSession = Depends(get_session)
) -> Response:
    """Delete a library book everywhere, including from every user's chats."""
    await documents.delete_document(session, document_id, user.id)
    return Response(status_code=204)
