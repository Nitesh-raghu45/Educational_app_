"""Conversation context: recent history, rolling summary, and standalone query rewriting."""

import logging
import re
import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..models import ChatMessage, ChatSection
from . import providers
from .web_images import WEB_REQUEST

logger = logging.getLogger(__name__)

MAX_HISTORY_CHARS = 1500
OLDER_HISTORY_CHARS = 500


def _clip(text: str, limit: int = MAX_HISTORY_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + " …"


# Markers in earlier answers point at THAT answer's sources and pictures. Left in, the model
# copies them into new answers, where the same numbers mean different sources or pictures.
_MARKER_RE = re.compile(
    r"\s*(?:\[\[\s*(?:figure|web)\s*:\s*\d{1,2}\s*\]\]|[\[【]\d{1,2}(?:†[^\]】]*)?[\]】])",
    re.IGNORECASE,
)
# Diagram and chart code is long and adds nothing as context; the text around it explains it.
_VISUAL_BLOCK_RE = re.compile(r"```[ \t]*(mermaid|chart)\b[^\n]*\n[\s\S]*?(?:```|$)", re.IGNORECASE)


def _context_text(message: ChatMessage) -> str:
    """A message as conversation context: answers without stale markers or visual code."""
    if message.role != "assistant":
        return message.content
    text = _VISUAL_BLOCK_RE.sub(lambda m: f"({'diagram' if m.group(1).lower() == 'mermaid' else 'chart'} shown)", message.content)
    return re.sub(r"\n{3,}", "\n\n", _MARKER_RE.sub("", text)).strip()


async def recent_history(
    session: AsyncSession, section_id: uuid.UUID, before: datetime, limit: int
) -> list[ChatMessage]:
    rows = await session.scalars(
        select(ChatMessage)
        .where(
            ChatMessage.section_id == section_id,
            ChatMessage.created_at < before,
            ChatMessage.status == "completed",
        )
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
    )
    return list(reversed(list(rows)))


def history_as_chat(history: list[ChatMessage]) -> list[dict[str, str]]:
    # The latest exchange in full (follow-ups mostly refer to it); older messages shortened, since
    # every token counts against the answer model's per-minute budget.
    last = len(history) - 2
    return [
        {"role": m.role, "content": _clip(_context_text(m), MAX_HISTORY_CHARS if i >= last else OLDER_HISTORY_CHARS)}
        for i, m in enumerate(history)
    ]


def _transcript(history: list[ChatMessage]) -> str:
    return "\n".join(
        f"{'Student' if m.role == 'user' else 'Assistant'}: {_clip(_context_text(m), 600)}" for m in history
    )


_WEB_REQUEST_RE = re.compile(WEB_REQUEST, re.IGNORECASE)

# Words that point back at the conversation ("explain it", "what about the second one?").
_REFERENCE_RE = re.compile(
    r"\b(?:it|its|it's|this|that|these|those|they|them|their|theirs|he|him|his|she|her|hers|one|ones"
    r"|former|latter|above|previous|earlier|again|more|else|same|other|another|second|third|last|next)\b",
    re.IGNORECASE,
)
# Questions that only make sense as a continuation ("and the stroma?", "why?").
_CONTINUATION_RE = re.compile(r"^\s*(?:and|but|or|so|also|then|why|how come|what about|how about)\b", re.IGNORECASE)


def needs_rewrite(question: str) -> bool:
    """Whether a question may depend on the conversation. A clearly standalone question skips the
    rewrite call (a full LLM round trip before search) because rewriting it wouldn't change it."""
    return (
        len(question.split()) <= 3
        or bool(_REFERENCE_RE.search(question))
        or bool(_CONTINUATION_RE.search(question))
        # "ok then find out on the internet": the topic is in the conversation, not the message.
        or bool(_WEB_REQUEST_RE.search(question))
    )


async def rewrite_query(summary: str | None, history: list[ChatMessage], question: str) -> str:
    """Turn a follow-up like "explain that again" into a self-contained search query."""
    if not history and not summary:
        return question
    if not needs_rewrite(question):
        return question
    context = ""
    if summary:
        context += f"Summary of earlier conversation:\n{summary}\n\n"
    if history:
        context += f"Recent conversation:\n{_transcript(history[-6:])}\n\n"
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the student's latest message as a standalone search query for their study "
                "documents. Resolve pronouns and references using the conversation. If it is already "
                "standalone, return it unchanged. Output only the query, no quotes or explanation."
            ),
        },
        {"role": "user", "content": f"{context}Latest message: {question}"},
    ]
    try:
        rewritten = await providers.llm().complete(
            messages, model=get_settings().groq_fast_model, temperature=0.0, max_tokens=1024
        )
    except Exception:
        logger.warning("Query rewrite failed; using the original question", exc_info=True)
        return question
    rewritten = rewritten.strip().strip('"').strip()
    return rewritten if 0 < len(rewritten) <= 500 else question


async def update_summary(session: AsyncSession, section: ChatSection) -> None:
    """Fold messages that have left the recent-history window into the section's rolling summary."""
    settings = get_settings()
    messages = list(
        await session.scalars(
            select(ChatMessage)
            .where(ChatMessage.section_id == section.id, ChatMessage.status == "completed")
            .order_by(ChatMessage.created_at)
        )
    )
    cutoff = len(messages) - settings.history_window
    if cutoff - section.summary_message_count < settings.summary_min_new_messages:
        return
    to_fold = messages[section.summary_message_count : cutoff]
    prompt = (
        "Update the running summary of a tutoring conversation. Keep the topics covered, key facts "
        "the student learned, and anything they said they struggle with. Max 200 words.\n\n"
        f"Current summary:\n{section.summary or '(none)'}\n\nNew messages:\n{_transcript(to_fold)}"
    )
    summary = await providers.llm().complete(
        [{"role": "user", "content": prompt}], model=settings.groq_fast_model, temperature=0.0, max_tokens=2048
    )
    if summary.strip():
        section.summary = summary.strip()
        section.summary_message_count = cutoff
        await session.commit()
