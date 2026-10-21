import uuid

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..db import get_session
from ..errors import AppError
from ..models import ChatMessage, ChatSection, User
from ..ratelimit import enforce_rate_limit
from ..schemas import (
    AttachDocument,
    DocumentOut,
    MessageCreate,
    MessageOut,
    QuizCreate,
    QuizOut,
    SectionCreate,
    SectionOut,
    SectionUpdate,
)
from ..security import current_user
from ..services import documents, quiz, rag, sections
from .quizzes import quiz_out
from .streaming import replay_response, start_generation, stream_response

router = APIRouter(prefix="/sections", tags=["sections"])


def _start_payload(section: ChatSection, user_message: ChatMessage, assistant_message: ChatMessage) -> dict:
    return {
        "section": SectionOut.model_validate(section).model_dump(mode="json"),
        "user_message": MessageOut.model_validate(user_message).model_dump(mode="json"),
        "assistant_message": MessageOut.model_validate(assistant_message).model_dump(mode="json"),
    }


@router.post("", response_model=SectionOut, status_code=201)
async def create_section(
    body: SectionCreate, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
) -> ChatSection:
    return await sections.create_section(session, user.id, body.title, get_settings().max_questions_per_section)


@router.get("", response_model=list[SectionOut])
async def list_sections(
    user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
) -> list[ChatSection]:
    return await sections.list_sections(session, user.id)


@router.get("/{section_id}", response_model=SectionOut)
async def get_section(
    section_id: uuid.UUID, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
) -> ChatSection:
    return await sections.get_owned_section(session, section_id, user.id)


@router.patch("/{section_id}", response_model=SectionOut)
async def rename_section(
    section_id: uuid.UUID,
    body: SectionUpdate,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ChatSection:
    section = await sections.get_owned_section(session, section_id, user.id)
    section.title = body.title
    await session.commit()
    return section


@router.delete("/{section_id}", status_code=204)
async def delete_section(
    section_id: uuid.UUID, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
) -> Response:
    await sections.delete_section(session, section_id, user.id)
    return Response(status_code=204)


@router.get("/{section_id}/messages", response_model=list[MessageOut])
async def list_messages(
    section_id: uuid.UUID, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
) -> list[ChatMessage]:
    return await sections.list_messages(session, section_id, user.id)


@router.post("/{section_id}/messages", response_class=StreamingResponse)
async def post_message(
    section_id: uuid.UUID,
    body: MessageCreate,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Submit a question and stream the answer as SSE: message_start, token*, citations, done | error."""
    await enforce_rate_limit(user.id, "messages", get_settings().rate_limit_per_minute)
    sub = await sections.submit_question(
        session, section_id=section_id, user_id=user.id, content=body.content, idempotency_key=body.client_message_id
    )
    start = _start_payload(sub.section, sub.user_message, sub.assistant_message)
    if sub.replayed:
        answer = sub.assistant_message
        if answer.status == "completed":
            return replay_response(start, answer.content, answer.citations)
        code, message = (
            ("ANSWER_FAILED", "This question's answer failed; retry it instead.")
            if answer.status == "failed"
            else ("ANSWER_IN_PROGRESS", "This question is still being answered.")
        )
        raise AppError(409, code, message, {"assistant_message_id": str(answer.id)})
    context = rag.AnswerContext(
        section_id=sub.section.id,
        user_id=sub.section.user_id,
        summary=sub.section.summary,
        question=sub.user_message.content,
        question_created_at=sub.user_message.created_at,
        document_ids=sub.document_ids or [],
    )
    return stream_response(start, start_generation(sub.assistant_message.id, context))


@router.post("/{section_id}/retry/{message_id}", response_class=StreamingResponse)
async def retry_answer(
    section_id: uuid.UUID,
    message_id: uuid.UUID,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Regenerate a failed answer. Does not consume another question slot."""
    await enforce_rate_limit(user.id, "messages", get_settings().rate_limit_per_minute)
    answer = await sections.reset_for_retry(session, section_id, user.id, message_id)
    section = await sections.get_owned_section(session, section_id, user.id)
    question = await session.get(ChatMessage, answer.reply_to_id)
    assert question is not None
    return stream_response(_start_payload(section, question, answer), start_generation(answer.id))


@router.get("/{section_id}/documents", response_model=list[DocumentOut])
async def list_section_documents(
    section_id: uuid.UUID, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    return await documents.section_documents(session, section_id, user.id)


@router.post("/{section_id}/documents", response_model=DocumentOut, status_code=201)
async def attach_document(
    section_id: uuid.UUID,
    body: AttachDocument,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    return await documents.attach(session, section_id, body.document_id, user.id)


@router.delete("/{section_id}/documents/{document_id}", status_code=204)
async def detach_document(
    section_id: uuid.UUID,
    document_id: uuid.UUID,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> Response:
    await documents.detach(session, section_id, document_id, user.id)
    return Response(status_code=204)


@router.post("/{section_id}/quizzes", response_model=QuizOut, status_code=201)
async def create_quiz(
    section_id: uuid.UUID,
    body: QuizCreate,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> QuizOut:
    await enforce_rate_limit(user.id, "quizzes", 5)
    section = await sections.get_owned_section(session, section_id, user.id)
    return quiz_out(await quiz.create_quiz(session, section, body.num_questions, body.topic))
