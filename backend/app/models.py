import uuid
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def utcnow() -> datetime:
    """Naive UTC timestamp (MariaDB DATETIME has no time zone)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# Microsecond precision so a question and its answer created in the same second keep their order.
Timestamp = DateTime().with_variant(mysql.DATETIME(fsp=6), "mysql", "mariadb")


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)


class ChatSection(Base):
    """One chat thread in the sidebar. Accepts at most `max_questions` user questions."""

    __tablename__ = "chat_sections"
    __table_args__ = (Index("idx_sections_user_updated", "user_id", "updated_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str | None] = mapped_column(String(200))
    question_count: Mapped[int] = mapped_column(Integer, default=0)
    max_questions: Mapped[int] = mapped_column(Integer, default=30)
    # active | limit_reached
    status: Mapped[str] = mapped_column(String(20), default="active")
    # Rolling summary of the messages that have scrolled out of the recent-history window.
    summary: Mapped[str | None] = mapped_column(Text)
    summary_message_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        Index("idx_messages_section_created", "section_id", "created_at"),
        UniqueConstraint("section_id", "idempotency_key", name="uq_messages_idempotency"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    section_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("chat_sections.id", ondelete="CASCADE"))
    # user | assistant
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text, default="")
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    # pending | streaming | completed | failed
    status: Mapped[str] = mapped_column(String(20), default="completed")
    # For assistant messages: the user question being answered.
    reply_to_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("chat_messages.id", ondelete="CASCADE"))
    idempotency_key: Mapped[str | None] = mapped_column(String(100))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (Index("idx_documents_user_created", "user_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(Integer)
    storage_path: Mapped[str] = mapped_column(String(500))
    # pending | processing | ready | failed
    status: Mapped[str] = mapped_column(String(20), default="pending")
    error: Mapped[str | None] = mapped_column(Text)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)


class DocumentChunk(Base):
    """Chunk text lives here; Pinecone holds only the vector plus filter metadata. `id` is the vector id."""

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    page: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)


class DailyUsage(Base):
    """How many of each costly action (question, quiz, upload, chat) a user did on a UTC day."""

    __tablename__ = "daily_usage"

    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    kind: Mapped[str] = mapped_column(String(20), primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0)


class DocumentEmbedding(Base):
    """Which embedding model a document's vectors were made with. When the configured model
    changes, documents without a row for it are re-embedded at startup. (A separate table, so
    adding it needs no migration of existing tables.)"""

    __tablename__ = "document_embeddings"

    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(String(200), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)


class DocumentFigure(Base):
    """A figure (diagram, chart, photo) from a document. Its caption is also stored as a
    DocumentChunk with id `chunk_id`, so search finds figures like any other passage."""

    __tablename__ = "document_figures"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_id: Mapped[str] = mapped_column(String(80), unique=True)
    page: Mapped[int | None] = mapped_column(Integer)
    storage_key: Mapped[str] = mapped_column(String(300))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(200))
    caption: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)


class FigureScan(Base):
    """Whether a document's figures were extracted (one row per document): drives the
    "N figures" label and the one-time backfill for documents uploaded before figures existed."""

    __tablename__ = "figure_scans"

    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    # processing | done | failed
    status: Mapped[str] = mapped_column(String(20), default="processing")
    figure_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)


class DocumentHash(Base):
    """SHA-256 of each uploaded file, so uploading the same file again reuses its document instead
    of processing a duplicate. A table of its own (not a column on documents): the app creates
    missing tables at startup but has no migrations to add columns to existing ones."""

    __tablename__ = "document_hashes"
    __table_args__ = (Index("idx_document_hashes_user_sha", "user_id", "sha256"),)

    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="CASCADE"))
    sha256: Mapped[str] = mapped_column(String(64))


class SectionDocument(Base):
    """Which documents a section is allowed to retrieve from."""

    __tablename__ = "section_documents"

    section_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("chat_sections.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)


class Quiz(Base):
    __tablename__ = "quizzes"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    section_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("chat_sections.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="CASCADE"))
    topic: Mapped[str | None] = mapped_column(String(200))
    # [{question, options, answer_index, explanation, citation}]
    questions: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)


class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    quiz_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("quizzes.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="CASCADE"))
    answers: Mapped[list[int | None]] = mapped_column(JSON)
    score: Mapped[int] = mapped_column(Integer)
    total: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)
