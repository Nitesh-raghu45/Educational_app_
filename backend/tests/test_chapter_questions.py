"""Asking for a chapter's questions ("important questions", "exercise questions") lists them all."""

import asyncio
import uuid

import pytest

from app.services.rag import (
    QUESTION_RULES,
    CHAPTER_RULES,
    build_messages,
    document_pictures_note,
    is_questions_request,
    question_evidence,
)
from app.services.retrieval import Evidence, question_score


@pytest.mark.parametrize(
    "text",
    [
        "Now show the important question in this chapter",
        "is this only one question in this chapter",
        "give me the exercise questions",
        "NCERT back exercise",
        "previous year questions",
        "imp questions of circles",
        "all questions from this chapter",
        "Exercise 10.2 solve karo",
    ],
)
def test_requests_for_the_chapters_questions_are_recognised(text):
    assert is_questions_request(text)


@pytest.mark.parametrize(
    "text",
    ["answer my previous question", "explain tangents", "what is exercise good for", "what is a question mark"],
)
def test_ordinary_questions_are_not(text):
    assert not is_questions_request(text)


def test_question_sections_score_above_explanations():
    exercises = (
        "EXERCISES\n1. What is the need for a system of control and coordination?\n"
        "2. Draw the structure of a neuron and explain its function.\n3. How does phototropism occur?"
    )
    explanation = "A tangent touches the circle at one point. Why does this happen? Because the radius is perpendicular."
    assert question_score(exercises) >= 5
    assert question_score(explanation) < 2


def _ev(n: int, text: str, score: float) -> Evidence:
    return Evidence(f"c{n}", uuid.UUID(int=1), "ch.pdf", n, text, score, "questions")


def test_question_evidence_keeps_reading_order_and_drops_least_question_like_first():
    chunks = [_ev(1, "a" * 400, 3), _ev(2, "b" * 400, 1), _ev(3, "c" * 400, 9)]
    kept = question_evidence(chunks, [], chars=850)
    assert [e.chunk_id for e in kept] == ["c1", "c3"]


def test_question_evidence_falls_back_to_search_matches():
    hit = _ev(7, "hit", 0.8)
    assert question_evidence([], [hit], chars=1000) == [hit]


def test_question_rules_replace_whole_chapter_rules():
    messages = build_messages(None, [], [_ev(1, "1. What is a tangent?", 5)], "important questions", "important questions",
                              whole_chapter=True, chapter_questions=True)
    system = messages[0]["content"]
    assert QUESTION_RULES.strip() in system and CHAPTER_RULES.strip() not in system


def test_web_pictures_offer_needs_a_picture_request():
    # "in this chapter" alone used to add "Want pictures of this from the web instead?".
    note = asyncio.run(document_pictures_note(None, [], "show the important question in this chapter", "..."))
    assert note == ""


# ---------- new exam questions ----------

from app.services.rag import EXAM_RULES, exam_evidence, is_exam_request  # noqa: E402


@pytest.mark.parametrize(
    "text",
    [
        "can you generate  the question that can appear in exam and test",
        "make a test for this chapter",
        "likely exam questions",
        "questions that may come in exam",
        "sample paper",
        "give me practice questions",
        "important questions for exam",
        "exam me kya aayega",
        "prepare 10 MCQs",
    ],
)
def test_requests_for_new_exam_questions_are_recognised(text):
    assert is_exam_request(text)


@pytest.mark.parametrize(
    "text",
    [
        "generate answers for these questions",
        "write answers to the exercise questions",
        "important questions of this chapter",  # the chapter's own list
        "what is a test cross",
        "how is the exam pattern",
    ],
)
def test_other_requests_are_not_exam_requests(text):
    assert not is_exam_request(text)


def test_exam_evidence_covers_the_chapter_then_its_questions():
    spread = [_ev(n, "x" * 1000, 0) for n in range(1, 11)]
    asked = [_ev(20, "1. What is a tangent?" * 10, 6), _ev(21, "2. Prove that…" * 10, 4)]
    kept = exam_evidence(asked, spread, chars=6000)
    ids = [e.chunk_id for e in kept]
    assert ids[-2:] == ["c20", "c21"]  # the chapter's own questions are included
    assert len(ids) > 4 and sum(len(e.text) for e in kept) <= 6000  # and the chapter's content


def test_exam_rules_replace_the_other_chapter_rules():
    messages = build_messages(None, [], [_ev(1, "text", 0)], "make a test", "make a test",
                              whole_chapter=True, chapter_questions=False, exam=True)
    system = messages[0]["content"]
    assert EXAM_RULES.strip() in system and QUESTION_RULES.strip() not in system and CHAPTER_RULES.strip() not in system


async def test_question_modes_skip_the_usual_search(client, fake_llm, monkeypatch):
    from .conftest import register
    from .test_chat import ask, new_section, upload_ready_doc

    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id, text="EXERCISES\n1. What is photosynthesis?\n2. Why do plants need light?")

    async def no_search(*args, **kwargs):
        raise AssertionError("the chapter's questions come from the whole chapter, not a search")

    monkeypatch.setattr("app.services.rag.retrieve", no_search)
    for text in ("generate the questions that can appear in exam", "exercise questions"):
        response = await ask(client, headers, section_id, text)
        assert response.status_code == 200
        system = fake_llm.stream_calls[-1][0]["content"]
        assert (EXAM_RULES if "exam" in text else QUESTION_RULES).strip() in system
