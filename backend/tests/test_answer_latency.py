"""Fewer waits before an answer: database connections checked only when idle, and the question
embedded while it is being saved."""

import time
import uuid

from sqlalchemy import text

from app import db
from app.services import providers, retrieval

from .conftest import register
from .test_chat import new_section, upload_ready_doc


async def test_only_idle_connections_are_pinged(monkeypatch):
    pings = []
    original = db.engine.dialect.do_ping
    monkeypatch.setattr(db.engine.dialect, "do_ping", lambda connection: pings.append(1) or original(connection))

    async with db.engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
    async with db.engine.connect() as connection:  # just used: no ping
        await connection.execute(text("SELECT 1"))
    assert pings == []

    monkeypatch.setattr(time, "monotonic", lambda: 10**9)  # an hour of idling, and more
    async with db.engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
    assert pings == [1]  # the pooled connection was checked before reuse


async def test_question_is_embedded_while_it_is_saved(client, monkeypatch):
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)

    embedder = providers.embedder()
    calls = []
    original = embedder.embed_query
    monkeypatch.setattr(embedder, "embed_query", lambda query: calls.append(query) or original(query))
    question = "What do mitochondria produce?"
    response = await client.post(
        f"/api/v1/sections/{section_id}/messages",
        json={"content": question, "client_message_id": uuid.uuid4().hex},
        headers=headers,
    )
    assert response.status_code == 200
    assert calls == [question]  # embedded once, early, and that result was used
    assert question not in retrieval._prefetched
