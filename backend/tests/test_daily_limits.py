import uuid

from app.config import get_settings
from app.services import providers

from .conftest import parse_sse, register
from .test_chat import ask, new_section, upload_ready_doc


async def test_daily_question_limit_spans_all_chats(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "daily_question_limit", 2)
    headers = await register(client)
    first, second = await new_section(client, headers), await new_section(client, headers)
    await upload_ready_doc(client, headers, first)
    await upload_ready_doc(client, headers, second)

    assert parse_sse((await ask(client, headers, first, "What is photosynthesis?")).text)[-1][0] == "done"
    assert parse_sse((await ask(client, headers, second, "What is photosynthesis?")).text)[-1][0] == "done"
    # Opening a new chat doesn't get around the daily cap.
    blocked = await ask(client, headers, second, "What does photosynthesis produce?")
    assert blocked.status_code == 429
    error = blocked.json()["error"]
    assert error["code"] == "DAILY_LIMIT_REACHED" and error["kind"] == "question" and error["limit"] == 2
    # The blocked question didn't use a slot in the chat either.
    section = (await client.get(f"/api/v1/sections/{second}", headers=headers)).json()
    assert section["question_count"] == 1

    usage = (await client.get("/api/v1/usage", headers=headers)).json()
    assert usage["limits"]["question"] == {"used": 2, "limit": 2}


async def test_resubmitting_the_same_question_is_not_counted_twice(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "daily_question_limit", 1)
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)
    first = await ask(client, headers, section_id, "What is photosynthesis?", key="same-key-123")
    again = await ask(client, headers, section_id, "What is photosynthesis?", key="same-key-123")
    assert first.status_code == again.status_code == 200
    assert parse_sse(again.text)[-1][0] == "done"


async def test_limits_are_per_user(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "daily_chat_limit", 1)
    alice, bob = await register(client, "alice@example.com"), await register(client, "bob@example.com")
    assert (await client.post("/api/v1/sections", json={}, headers=alice)).status_code == 201
    blocked = await client.post("/api/v1/sections", json={}, headers=alice)
    assert blocked.status_code == 429 and blocked.json()["error"]["kind"] == "chat"
    assert (await client.post("/api/v1/sections", json={}, headers=bob)).status_code == 201


async def test_daily_upload_limit(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "daily_upload_limit", 1)
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)
    blocked = await client.post(
        "/api/v1/documents", files={"file": ("more.txt", b"More notes", "text/plain")},
        data={"section_id": section_id}, headers=headers,
    )
    assert blocked.status_code == 429 and blocked.json()["error"]["kind"] == "upload"


async def test_failed_quiz_is_refunded(client, monkeypatch, fake_llm):
    monkeypatch.setattr(get_settings(), "daily_quiz_limit", 1)
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)

    async def broken(*args, **kwargs):
        return "not json"

    monkeypatch.setattr(fake_llm, "complete", broken)
    failed = await client.post(f"/api/v1/sections/{section_id}/quizzes", json={"num_questions": 3}, headers=headers)
    assert failed.status_code == 502
    monkeypatch.undo()
    monkeypatch.setattr(get_settings(), "daily_quiz_limit", 1)
    ok = await client.post(f"/api/v1/sections/{section_id}/quizzes", json={"num_questions": 3}, headers=headers)
    assert ok.status_code == 201, "the failed quiz must not have used the only daily slot"
    blocked = await client.post(f"/api/v1/sections/{section_id}/quizzes", json={"num_questions": 3}, headers=headers)
    assert blocked.status_code == 429


async def test_zero_means_unlimited(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "daily_chat_limit", 0)
    headers = await register(client)
    for _ in range(3):
        assert (await client.post("/api/v1/sections", json={}, headers=headers)).status_code == 201
    assert (await client.get("/api/v1/usage", headers=headers)).json()["limits"]["chat"]["limit"] == 0
