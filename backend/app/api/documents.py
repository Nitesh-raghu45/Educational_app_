import logging
import time
import uuid

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..models import Document, User
from ..ratelimit import enforce_rate_limit
from ..schemas import DocumentOut
from ..security import current_user
from ..services import documents
from ..services.jobs import enqueue_ingestion

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    section_id: uuid.UUID | None = Form(None),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> Document:
    """Store the file privately and queue it for parsing, chunking and embedding."""
    started = time.perf_counter()
    await enforce_rate_limit(user.id, "uploads", 30)
    doc = await documents.save_upload(session, user.id, file, section_id)
    await enqueue_ingestion(doc.id)
    # The browser->server transfer happens before this handler runs (the progress bar shows it);
    # this covers saving the file and creating the database record.
    logger.info("Upload %s (%r, %d KB) saved in %.1fs", doc.id, doc.filename, doc.size_bytes // 1024, time.perf_counter() - started)
    return doc


@router.get("", response_model=list[DocumentOut])
async def list_documents(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    return await documents.list_documents(session, user.id)


@router.get("/{document_id}/status", response_model=DocumentOut)
async def document_status(
    document_id: uuid.UUID, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
) -> Document:
    return await documents.get_owned_document(session, document_id, user.id)


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
) -> Response:
    await documents.delete_document(session, document_id, user.id)
    return Response(status_code=204)
