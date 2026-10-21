"""Round-trip budget for asking a question.

On a host like Render, the database (Aiven) and Redis (Upstash) are remote, so every SQL
statement before the first token adds network latency. These tests pin the count so it can't
silently grow back.
"""

import asyncio

from sqlalchemy import event

from app.db import engine
from app.services import jobs

from .conftest import parse_sse, register
from .test_chat import ask, new_section, upload_ready_doc


class StatementCounter:
    def __init__(self):
        self.statements: list[str] = []

    def __call__(self, conn, cursor, statement, *args):
        self.statements.append(" ".join(statement.split())[:80])


async def _count_statements_for_question(client, question: str):
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)
    await ask(client, headers, section_id, "What is photosynthesis?")  # warm caches, add history
    await asyncio.gather(*list(jobs._tasks), return_exceptions=True)

    counter = StatementCounter()
    event.listen(engine.sync_engine, "before_cursor_execute", counter)
    try:
        response = await ask(client, headers, section_id, question)
        await asyncio.gather(*list(jobs._tasks), return_exceptions=True)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", counter)
    assert parse_sse(response.text)[-1][0] == "done"
    return counter.statements


async def test_question_round_trip_budget(client):
    statements = await _count_statements_for_question(client, "What does photosynthesis produce?")
    # Was 18 before: an up-front idempotency lookup, re-reading the question/section/answer and
    # document list in the generation task, a "streaming" status write, and two separate INSERTs.
    assert len(statements) <= 12, "\n".join(statements)


async def test_standalone_follow_up_skips_rewrite_but_real_follow_up_keeps_it(client, fake_llm, monkeypatch):
    from app.services.conversation import needs_rewrite

    assert not needs_rewrite("Where does photosynthesis take place?")
    assert needs_rewrite("What does it produce?") and needs_rewrite("and the stroma?") and needs_rewrite("why?")

    calls = []
    original = fake_llm.complete

    async def counting_complete(messages, model, **kwargs):
        calls.append(messages)
        return await original(messages, model, **kwargs)

    monkeypatch.setattr(fake_llm, "complete", counting_complete)
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)
    await ask(client, headers, section_id, "What is photosynthesis?")
    await ask(client, headers, section_id, "Where does photosynthesis produce oxygen?")
    assert calls == [], "a standalone question must not wait for a rewrite call"
    await ask(client, headers, section_id, "What does it produce?")
    assert len(calls) == 1, "a real follow-up is still rewritten"
