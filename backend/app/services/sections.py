"""Section lifecycle and the per-section question quota.

The quota is enforced in MariaDB, never in the client or Redis: a conditional UPDATE reserves a
slot atomically (InnoDB row lock), and the user message is inserted in the same transaction.
MariaDB has no UPDATE ... RETURNING, so we check rowcount and read the new count afterwards.
"""

import uuid
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import and_, delete, insert, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..errors import AppError
from . import usage
from ..models import ChatMessage, ChatSection, Document, Quiz, QuizAttempt, SectionDocument, utcnow

# A pending/streaming answer older than this is assumed orphaned (e.g. the server restarted).
STALE_GENERATION = timedelta(minutes=2)


@dataclass
class Submission:
    section: ChatSection
    user_message: ChatMessage
    assistant_message: ChatMessage
    # True when the idempotency key matched an earlier submission; no slot was reserved.
    replayed: bool = False
    # Ready documents found during the pre-check, handed to generation so it needn't query again.
    document_ids: list[uuid.UUID] | None = None


def section_not_found() -> AppError:
    # Same response for "doesn't exist" and "belongs to someone else".
    return AppError(404, "SECTION_NOT_FOUND", "This chat does not exist.")


async def get_owned_section(session: AsyncSession, section_id: uuid.UUID, user_id: uuid.UUID) -> ChatSection:
    section = await session.scalar(
        select(ChatSection).where(ChatSection.id == section_id, ChatSection.user_id == user_id)
    )
    if section is None:
        raise section_not_found()
    return section


async def create_section(
    session: AsyncSession, user_id: uuid.UUID, title: str | None, max_questions: int
) -> ChatSection:
    await usage.reserve(session, user_id, "chat")
    section = ChatSection(id=uuid.uuid4(), user_id=user_id, title=title, max_questions=max_questions)
    session.add(section)
    await session.commit()
    return section


async def list_sections(session: AsyncSession, user_id: uuid.UUID, limit: int = 200) -> list[ChatSection]:
    rows = await session.scalars(
        select(ChatSection)
        .where(ChatSection.user_id == user_id)
        .order_by(ChatSection.updated_at.desc())
        .limit(limit)
    )
    return list(rows)


async def delete_section(session: AsyncSession, section_id: uuid.UUID, user_id: uuid.UUID) -> None:
    await get_owned_section(session, section_id, user_id)
    quiz_ids = select(Quiz.id).where(Quiz.section_id == section_id)
    await session.execute(delete(QuizAttempt).where(QuizAttempt.quiz_id.in_(quiz_ids)))
    await session.execute(delete(Quiz).where(Quiz.section_id == section_id))
    await session.execute(delete(SectionDocument).where(SectionDocument.section_id == section_id))
    # Answers reference questions, so remove them first.
    await session.execute(
        delete(ChatMessage).where(ChatMessage.section_id == section_id, ChatMessage.role == "assistant")
    )
    await session.execute(delete(ChatMessage).where(ChatMessage.section_id == section_id))
    await session.execute(delete(ChatSection).where(ChatSection.id == section_id))
    await session.commit()


async def list_messages(session: AsyncSession, section_id: uuid.UUID, user_id: uuid.UUID) -> list[ChatMessage]:
    await get_owned_section(session, section_id, user_id)
    rows = await session.scalars(
        select(ChatMessage).where(ChatMessage.section_id == section_id).order_by(ChatMessage.created_at)
    )
    return list(rows)


async def ready_document_ids(session: AsyncSession, section_id: uuid.UUID) -> list[uuid.UUID]:
    rows = await session.scalars(
        select(Document.id)
        .join(SectionDocument, SectionDocument.document_id == Document.id)
        .where(SectionDocument.section_id == section_id, Document.status == "ready")
    )
    return list(rows)


def _title_from(question: str) -> str:
    title = " ".join(question.split())
    return title if len(title) <= 60 else title[:57].rstrip() + "..."


def _closed_error(section: ChatSection) -> AppError:
    if section.status == "limit_reached" or section.question_count >= section.max_questions:
        return AppError(
            409,
            "SECTION_LIMIT_REACHED",
            f"This chat has reached its {section.max_questions}-question limit. Start a new chat to continue.",
        )
    return AppError(409, "SECTION_CLOSED", "This chat is closed to new questions.")


async def _find_submission(
    session: AsyncSession, section_id: uuid.UUID, user_id: uuid.UUID, key: str
) -> Submission | None:
    row = (
        await session.execute(
            select(ChatMessage, ChatSection)
            .join(ChatSection, ChatSection.id == ChatMessage.section_id)
            .where(
                ChatMessage.section_id == section_id,
                ChatSection.user_id == user_id,
                ChatMessage.idempotency_key == key,
                ChatMessage.role == "user",
            )
        )
    ).first()
    if row is None:
        return None
    user_message, section = row
    assistant = await session.scalar(
        select(ChatMessage)
        .where(ChatMessage.reply_to_id == user_message.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(1)
    )
    if assistant is None:
        return None
    return Submission(section, user_message, assistant, replayed=True)


async def submit_question(
    session: AsyncSession,
    *,
    section_id: uuid.UUID,
    user_id: uuid.UUID,
    content: str,
    idempotency_key: str | None,
) -> Submission:
    """Reserve one question slot and persist the question plus a pending answer, atomically.

    Duplicate submissions (same idempotency key) are detected by the unique index when the
    question is inserted, not by an extra lookup up front: one fewer database round trip for
    every normal question.
    """

    async def replay_or_raise(error: AppError) -> Submission:
        # A retried request for a question that was already accepted (maybe the 30th) replays it.
        if idempotency_key and (existing := await _find_submission(session, section_id, user_id, idempotency_key)):
            return existing
        raise error

    # Friendly pre-checks; the conditional UPDATE below is what actually enforces the limit.
    section = await get_owned_section(session, section_id, user_id)
    if section.status != "active" or section.question_count >= section.max_questions:
        return await replay_or_raise(_closed_error(section))
    document_ids = await ready_document_ids(session, section_id)
    if not document_ids:
        raise AppError(
            409, "NO_READY_DOCUMENTS", "Upload or attach at least one processed document to this chat first."
        )

    # Daily per-user cap; rolled back with everything else if the question isn't accepted.
    # A resubmission of a question accepted earlier still gets its answer, even at the limit.
    try:
        await usage.reserve(session, user_id, "question")
    except AppError as error:
        return await replay_or_raise(error)

    now = utcnow()
    result = await session.execute(
        update(ChatSection)
        .where(
            ChatSection.id == section_id,
            ChatSection.user_id == user_id,
            ChatSection.status == "active",
            ChatSection.question_count < ChatSection.max_questions,
        )
        .values(question_count=ChatSection.question_count + 1, updated_at=now)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        await session.rollback()
        return await replay_or_raise(_closed_error(await get_owned_section(session, section_id, user_id)))

    # Our UPDATE holds the row lock until commit, so this read is consistent.
    await session.refresh(section)
    if section.question_count >= section.max_questions:
        section.status = "limit_reached"
    if not section.title:
        section.title = _title_from(content)

    question_row = dict(
        id=uuid.uuid4(),
        section_id=section_id,
        role="user",
        content=content,
        citations=None,
        status="completed",
        reply_to_id=None,
        idempotency_key=idempotency_key,
        error=None,
        created_at=now,
        updated_at=now,
    )
    answer_row = dict(
        question_row,
        id=uuid.uuid4(),
        role="assistant",
        content="",
        status="pending",
        reply_to_id=question_row["id"],
        idempotency_key=None,
        created_at=now + timedelta(microseconds=1),
    )
    try:
        # Both rows in one multi-row INSERT (one round trip); the question comes first for the FK.
        await session.execute(insert(ChatMessage.__table__), [question_row, answer_row])
        await session.commit()
    except IntegrityError:
        # A concurrent request with the same idempotency key won; our slot reservation rolls back too.
        await session.rollback()
        if idempotency_key and (existing := await _find_submission(session, section_id, user_id, idempotency_key)):
            return existing
        raise
    return Submission(section, ChatMessage(**question_row), ChatMessage(**answer_row), document_ids=document_ids)


async def reset_for_retry(
    session: AsyncSession, section_id: uuid.UUID, user_id: uuid.UUID, message_id: uuid.UUID
) -> ChatMessage:
    """Put a failed (or orphaned) answer back to pending. Never touches the question quota."""
    await get_owned_section(session, section_id, user_id)
    now = utcnow()
    result = await session.execute(
        update(ChatMessage)
        .where(
            ChatMessage.id == message_id,
            ChatMessage.section_id == section_id,
            ChatMessage.role == "assistant",
            or_(
                ChatMessage.status == "failed",
                and_(
                    ChatMessage.status.in_(["pending", "streaming"]),
                    ChatMessage.updated_at < now - STALE_GENERATION,
                ),
            ),
        )
        .values(status="pending", content="", citations=None, error=None, updated_at=now)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        await session.rollback()
        message = await session.get(ChatMessage, message_id)
        if message is None or message.section_id != section_id or message.role != "assistant":
            raise AppError(404, "MESSAGE_NOT_FOUND", "That answer does not exist.")
        raise AppError(409, "MESSAGE_NOT_RETRYABLE", "Only failed answers can be retried.")
    await session.commit()
    message = await session.get(ChatMessage, message_id, populate_existing=True)
    assert message is not None
    return message
