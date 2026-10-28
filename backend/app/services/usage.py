"""Daily per-user limits on the actions that cost money (LLM calls, embedding work).

The per-chat 30-question limit alone doesn't cap cost, because a user can open unlimited chats.
These limits are enforced in the database the same way as that quota: a conditional UPDATE
(`count < limit`) inside the caller's transaction, so concurrent requests can't overshoot and
a failed action rolls its reservation back. Counters reset at midnight UTC.
"""

import uuid
from datetime import date, datetime, time, timedelta, timezone

from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..errors import AppError
from ..models import DailyUsage

KINDS = ("question", "quiz", "upload", "chat")
_LABELS = {
    "question": "questions",
    "quiz": "quizzes",
    "upload": "document uploads",
    "chat": "new chats",
    "figure": "figure descriptions",
}


def limit_for(kind: str) -> int:
    """0 means unlimited."""
    settings = get_settings()
    return {
        "question": settings.daily_question_limit,
        "quiz": settings.daily_quiz_limit,
        "upload": settings.daily_upload_limit,
        "chat": settings.daily_chat_limit,
        # Not shown in the usage meter: when reached, new uploads just get no more figures today.
        "figure": settings.daily_figure_limit,
    }[kind]


def today() -> date:
    return datetime.now(timezone.utc).date()


def resets_at() -> datetime:
    return datetime.combine(today() + timedelta(days=1), time.min, tzinfo=timezone.utc)


async def _ensure_row(session: AsyncSession, user_id: uuid.UUID, kind: str) -> None:
    """Create today's counter if it doesn't exist (in its own small transaction)."""
    exists = await session.scalar(
        select(DailyUsage.count).where(DailyUsage.user_id == user_id, DailyUsage.day == today(), DailyUsage.kind == kind)
    )
    if exists is not None:
        return
    try:
        await session.execute(insert(DailyUsage).values(user_id=user_id, day=today(), kind=kind, count=0))
        await session.commit()
    except IntegrityError:  # a concurrent request created it first
        await session.rollback()


async def _try_increment(session: AsyncSession, user_id: uuid.UUID, kind: str, limit: int) -> bool:
    result = await session.execute(
        update(DailyUsage)
        .where(
            DailyUsage.user_id == user_id,
            DailyUsage.day == today(),
            DailyUsage.kind == kind,
            DailyUsage.count < limit,
        )
        .values(count=DailyUsage.count + 1)
        .execution_options(synchronize_session=False)
    )
    return result.rowcount == 1


async def reserve(session: AsyncSession, user_id: uuid.UUID, kind: str) -> None:
    """Count one action against today's limit, or raise 429 DAILY_LIMIT_REACHED.

    Does not commit: the reservation becomes permanent together with the caller's action, and
    disappears if the caller rolls back.
    """
    limit = limit_for(kind)
    if limit <= 0:
        return
    # Usual case: today's counter exists, so this is one round trip.
    if await _try_increment(session, user_id, kind, limit):
        return
    # First action of the day (no counter yet), or the limit is reached.
    await _ensure_row(session, user_id, kind)
    if not await _try_increment(session, user_id, kind, limit):
        await session.rollback()
        raise AppError(
            429,
            "DAILY_LIMIT_REACHED",
            f"You've reached today's limit of {limit} {_LABELS[kind]}. It resets at midnight UTC.",
            {"kind": kind, "limit": limit, "resets_at": resets_at().isoformat()},
        )


async def refund(session: AsyncSession, user_id: uuid.UUID, kind: str) -> None:
    """Give back a reservation whose action failed after it was committed (e.g. quiz generation)."""
    if limit_for(kind) <= 0:
        return
    await session.execute(
        update(DailyUsage)
        .where(DailyUsage.user_id == user_id, DailyUsage.day == today(), DailyUsage.kind == kind, DailyUsage.count > 0)
        .values(count=DailyUsage.count - 1)
        .execution_options(synchronize_session=False)
    )
    await session.commit()


async def get_usage(session: AsyncSession, user_id: uuid.UUID) -> dict:
    rows = dict(
        (
            await session.execute(
                select(DailyUsage.kind, DailyUsage.count).where(DailyUsage.user_id == user_id, DailyUsage.day == today())
            )
        ).all()
    )
    return {
        "resets_at": resets_at().isoformat(),
        "limits": {kind: {"used": rows.get(kind, 0), "limit": limit_for(kind)} for kind in KINDS},
    }
