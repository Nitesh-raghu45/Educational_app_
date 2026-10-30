"""Switching to the fallback model when the answer model is busy, and a smaller prompt."""

from app.config import get_settings
from app.models import ChatMessage
from app.services import conversation
from app.services.llm import ModelBusy
from app.services.rag import build_messages
from app.services.retrieval import Evidence
from app.services.web_images import WebImage

from .conftest import parse_sse, register
from .test_chat import ask, new_section, upload_ready_doc


def busy_answer_model(fake_llm, busy_after_tokens=False):
    calls = []

    async def stream(messages, model, **kwargs):
        calls.append((model, kwargs.get("retry", True)))
        if model == get_settings().groq_answer_model:
            if busy_after_tokens:
                yield "Plants "
            raise ModelBusy(f"{model}: HTTP 429")
        yield "Photosynthesis makes glucose [1]."

    fake_llm.stream = stream
    return calls


async def test_busy_answer_model_switches_to_fallback_at_once(client, fake_llm):
    calls = busy_answer_model(fake_llm)
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)

    events = parse_sse((await ask(client, headers, section_id, "What does photosynthesis produce?")).text)
    assert "".join(d["text"] for n, d in events if n == "token") == "Photosynthesis makes glucose [1]."
    assert [n for n, _ in events][-1] == "done"
    settings = get_settings()
    # No client-side retries (waiting out the limit) on the answer model; then the fallback.
    assert calls == [(settings.groq_answer_model, False), (settings.groq_fallback_model, True)]


async def test_without_fallback_the_answer_model_retries(client, fake_llm, monkeypatch):
    monkeypatch.setattr(get_settings(), "groq_fallback_model", "")
    calls = busy_answer_model(fake_llm)
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)
    events = parse_sse((await ask(client, headers, section_id, "What does photosynthesis produce?")).text)
    assert calls == [(get_settings().groq_answer_model, True)]
    assert events[-1][0] == "error"


async def test_no_switch_once_the_answer_has_started(client, fake_llm):
    """Switching mid-answer would mix two answers, so a failure after the first word is an error."""
    calls = busy_answer_model(fake_llm, busy_after_tokens=True)
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)
    events = parse_sse((await ask(client, headers, section_id, "What does photosynthesis produce?")).text)
    assert len(calls) == 1 and events[-1][0] == "error"


def evidence(figure=None):
    return [Evidence("c1", "d1", "notes.pdf", 1, "Plants make glucose.", 0.9, figure=figure)]


def test_figure_and_web_rules_only_when_needed():
    plain = build_messages(None, [], evidence(), "q", "q")[0]["content"]
    assert "[[figure:N]]" not in plain and "[[web:N]]" not in plain

    with_figure = build_messages(None, [], evidence({"id": "f", "title": "t", "width": 1, "height": 1}), "q", "q")
    assert "[[figure:N]]" in with_figure[0]["content"] and "[[web:N]]" not in with_figure[0]["content"]

    image = WebImage("Heart", "https://a.org/h.png", "https://a.org/h", "a.org", "wikipedia")
    with_web = build_messages(None, [], evidence(), "q", "q", [image])[0]["content"]
    assert "[[web:N]]" in with_web and "[[figure:N]]" not in with_web


def test_older_history_is_shortened():
    long = "word " * 600
    history = [ChatMessage(role=r, content=long) for r in ("user", "assistant", "user", "assistant")]
    lengths = [len(m["content"]) for m in conversation.history_as_chat(history)]
    assert lengths[0] < 600 and lengths[1] < 600  # older exchange
    assert lengths[2] > 1400 and lengths[3] > 1400  # the latest exchange, in full
