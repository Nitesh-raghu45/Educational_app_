"""The message router: small talk and reworking the previous answer skip the search."""

import logging

import pytest

from app.services import router
from app.services.rag import REWORK_PROMPT, NOT_FOUND_ANSWER

from .conftest import parse_sse, register
from .test_chat import ask, new_section, upload_ready_doc


@pytest.mark.parametrize(
    "text, kind",
    [
        ("hi", "greeting"),
        ("Hiii!!", "greeting"),
        ("hello sir", "greeting"),
        ("नमस्ते", "greeting"),
        ("thank you so much", "thanks"),
        ("ok thanks", "thanks"),
        ("dhanyawad bhai", "thanks"),
        ("धन्यवाद", "thanks"),
        ("okk", "ack"),
        ("got it 👍", "ack"),
        ("samajh gaya", "ack"),
        ("ठीक है", "ack"),
        ("bye", "bye"),
        ("what can you do?", "about"),
    ],
)
def test_small_talk_is_chat(text, kind):
    decision = router.route(text)
    assert (decision.route, decision.reason) == ("chat", kind)
    assert decision.reply


@pytest.mark.parametrize(
    "text, kind",
    [
        ("in hindi", "language"),
        ("hindi me samjhao", "language"),
        ("Hindi mein batao please", "language"),
        ("हिंदी में समझाओ", "language"),
        ("translate to hindi", "language"),
        ("make it shorter", "shorter"),
        ("explain it in simple words", "simpler"),
        ("can you put that in points", "format"),
        ("in a table please", "format"),
        ("explain again", "repeat"),
    ],
)
def test_reworking_the_previous_answer(text, kind):
    assert (router.route(text).route, router.route(text).reason) == ("rework", kind)


@pytest.mark.parametrize(
    "text",
    [
        "What is a tangent?",
        "hi, what is a tangent?",  # small talk plus a real question
        "thanks, now explain photosynthesis",
        "ok explain tangents",
        "explain tangents in hindi",  # names a topic: search for it
        "explain in more detail",  # wants more than the answer had
        "give examples",
        "summarise this chapter",
        "summarize it",  # could mean the chapter: keep the whole-chapter flow
        "what is it in hindi",
        "simple pendulum",
        "points of contact",
        "short circuit",
        "What is a table?",
        "yes",  # may accept an offer in the last answer
        "haan",
        "help me with chapter 10",
    ],
)
def test_everything_else_searches_the_documents(text):
    assert router.route(text).route == "documents"


def test_hindi_small_talk_gets_a_hindi_reply():
    assert router.route("नमस्ते").reply == router.CHAT_REPLIES_HINDI["greeting"]


async def test_greeting_is_answered_without_search_or_model(client, fake_llm, monkeypatch, caplog):
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)

    async def no_search(*args, **kwargs):
        raise AssertionError("small talk must not search")

    monkeypatch.setattr("app.services.rag.retrieve", no_search)
    with caplog.at_level(logging.INFO, logger="app.services.router"):
        response = await ask(client, headers, section_id, "thanks!")
    events = parse_sse(response.text)
    assert events[-1][0] == "done"
    assert "".join(d["text"] for n, d in events if n == "token") == router.CHAT_REPLIES["thanks"]
    assert fake_llm.stream_calls == []
    assert "route=chat reason=thanks" in caplog.text


async def test_rework_uses_previous_answer_without_search(client, fake_llm, monkeypatch, caplog):
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)
    await ask(client, headers, section_id, "What does photosynthesis produce?")

    async def no_search(*args, **kwargs):
        raise AssertionError("a rework must not search")

    monkeypatch.setattr("app.services.rag.retrieve", no_search)
    with caplog.at_level(logging.INFO, logger="app.services.router"):
        response = await ask(client, headers, section_id, "hindi me samjhao")
    events = parse_sse(response.text)
    assert events[-1][0] == "done"
    messages = fake_llm.stream_calls[-1]
    assert messages[0]["content"] == REWORK_PROMPT
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    # The previous answer, without its source numbers, which mean nothing in the new answer.
    assert messages[2]["content"] == "Photosynthesis turns light into chemical energy."
    assert messages[3]["content"] == "hindi me samjhao"
    assert "route=rework reason=language" in caplog.text


async def test_rework_without_previous_answer_searches(client, fake_llm, caplog):
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)
    with caplog.at_level(logging.INFO, logger="app.services.router"):
        await ask(client, headers, section_id, "in simple words")
    assert fake_llm.stream_calls[-1][0]["content"] != REWORK_PROMPT
    assert "route=documents reason=rework-simpler-without-previous-answer" in caplog.text


def test_refusals_and_small_talk_are_not_reworked():
    from types import SimpleNamespace as M

    from app.services.rag import previous_answer

    question = M(role="user", content="What is X?")
    assert previous_answer([question, M(role="assistant", content=NOT_FOUND_ANSWER)]) is None
    assert previous_answer([question, M(role="assistant", content=router.CHAT_REPLIES["thanks"])]) is None
    real = M(role="assistant", content="X is Y [1].")
    assert previous_answer([question, real]) == (question, real)
