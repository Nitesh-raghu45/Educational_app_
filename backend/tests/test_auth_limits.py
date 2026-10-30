"""Limits on signing up and in, answer reasoning effort, and clean conversation history."""

import asyncio
import time

from app.config import get_settings
from app.models import ChatMessage
from app.services import conversation
from app.services.rag import reasoning_effort_for

from .conftest import register


async def signup(client, email, ip="203.0.113.1"):
    return await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "password123"},
        headers={"CF-Connecting-IP": ip},
    )


async def login(client, email, password, ip="203.0.113.9"):
    return await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
        headers={"CF-Connecting-IP": ip},
    )


async def test_signups_limited_per_ip_and_app_wide(client, monkeypatch):
    for i in range(5):
        assert (await signup(client, f"s{i}@example.com")).status_code == 201
    blocked = await signup(client, "s5@example.com")
    assert blocked.status_code == 429 and blocked.json()["error"]["code"] == "RATE_LIMITED"
    # Another address can still sign up...
    assert (await signup(client, "other@example.com", ip="198.51.100.7")).status_code == 201

    # ...until the app-wide cap, which holds however many IPs are used.
    monkeypatch.setattr(get_settings(), "signup_limit_per_hour", 7)
    assert (await signup(client, "x1@example.com", ip="198.51.100.8")).status_code == 201
    assert (await signup(client, "x2@example.com", ip="198.51.100.9")).status_code == 429


async def test_password_guessing_limited_per_account(client):
    await register(client, "victim@example.com")
    for i in range(10):
        # A different IP every time: the per-account limit still applies.
        response = await login(client, "victim@example.com", f"wrong-password-{i}", ip=f"198.51.100.{i}")
        assert response.status_code == 401
    response = await login(client, "victim@example.com", "password123", ip="198.51.100.200")
    assert response.status_code == 429
    # Other accounts are unaffected.
    await register(client, "someone@example.com")
    assert (await login(client, "someone@example.com", "password123")).status_code == 200


async def test_password_checks_dont_block_other_requests(client):
    """bcrypt runs in a thread: while sign-ins are checked, other requests are still answered."""
    headers = await register(client, "busy@example.com")
    logins = [asyncio.create_task(login(client, "busy@example.com", "wrong-password", ip=f"192.0.2.{i}")) for i in range(4)]
    await asyncio.sleep(0)
    started = time.monotonic()
    assert (await client.get("/api/v1/auth/me", headers=headers)).status_code == 200
    me_took = time.monotonic() - started
    await asyncio.gather(*logins)
    assert me_took < 0.5


async def test_unknown_email_gets_the_same_answer(client):
    response = await login(client, "nobody@example.com", "password123")
    assert response.status_code == 401 and response.json()["error"]["code"] == "INVALID_CREDENTIALS"


def test_reasoning_effort():
    assert reasoning_effort_for("What is osmosis?", "What is osmosis?") == "low"
    assert reasoning_effort_for("Who discovered penicillin", "Who discovered penicillin") == "low"
    assert reasoning_effort_for("Why do leaves change colour?", "Why do leaves change colour?") == "medium"
    assert reasoning_effort_for("Compare mitosis and meiosis", "Compare mitosis and meiosis") == "medium"
    assert reasoning_effort_for("Summarize this document", "Summarize this document") == "medium"
    assert reasoning_effort_for("Draw a flowchart of digestion", "Draw a flowchart of digestion") == "medium"


def test_history_drops_stale_markers_and_visual_code():
    answer = ChatMessage(
        role="assistant",
        content=(
            "Plants make glucose [1][3]. Oxygen is released 【2†source】.\n\n[[figure:2]]\n\n"
            "```mermaid\nflowchart TD\nA-->B\n```\n\nSee the photo.\n\n[[web:1]]"
        ),
    )
    question = ChatMessage(role="user", content="What is [1] here?")
    chat = conversation.history_as_chat([question, answer])
    assert chat[0]["content"] == "What is [1] here?"  # the student's words are kept as typed
    assert chat[1]["content"] == "Plants make glucose. Oxygen is released.\n\n(diagram shown)\n\nSee the photo."
