import uuid

from app.services.rag import chapter_evidence, is_overview_question
from app.services.retrieval import Evidence, evenly_spaced

from .conftest import parse_sse, register
from .test_chat import ask, new_section, upload_ready_doc

# Deliberately shares no words with "what is this document about".
RESUME = (
    "Nitesh Raghuwanshi. Software engineer skilled in Python, FastAPI and React.\n\n"
    "Experience: built retrieval augmented generation systems and data pipelines.\n\n"
    "Education: Bachelor of Technology in computer science."
)


def test_overview_detection():
    for question in (
        "what is this document about", "Summarize my notes", "give me the key points", "What is in the PDF?",
        "i want to understand first this chapter", "explain chapter 6", "Teach me this lesson",
        "revise the whole unit", "go through the complete chapter", "yeh chapter samjhao",
    ):
        assert is_overview_question(question), question
    for question in (
        "what is photosynthesis", "Who won the 2018 football world cup?", "Explain enzymes",
        "explain the basic proportionality theorem", "how do I learn to prove similarity?",
    ):
        assert not is_overview_question(question), question


def _evidence(n, text="x" * 100):
    return Evidence(f"c{n}", uuid.uuid4(), "ch.pdf", n, text, 0.0, "overview")


def test_evenly_spaced_keeps_the_first_and_last():
    assert evenly_spaced(list(range(10)), 4) == [0, 3, 6, 9]
    assert evenly_spaced([1, 2], 5) == [1, 2]
    assert evenly_spaced([1, 2, 3], 0) == []


def test_chapter_evidence_fits_the_budget_across_the_whole_chapter():
    spread = [_evidence(n) for n in range(40)]
    hits = [_evidence(100 + n) for n in range(10)] + [spread[5]]
    chosen = chapter_evidence(hits, spread, max_hits=3, chars=2000)

    assert sum(len(e.text) for e in chosen) <= 2000
    assert [e.chunk_id for e in chosen[-3:]] == ["c100", "c101", "c102"]  # best matches kept
    pages = [e.page for e in chosen[:-3]]
    assert pages == sorted(pages) and pages[0] == 0 and pages[-1] == 39  # start to end, in order


async def test_whole_chapter_question_gets_passages_from_all_through_it(client, fake_llm):
    headers = await register(client)
    section_id = await new_section(client, headers)
    parts = [f"Part {n}: idea number {n} of the chapter, explained with care." for n in range(60)]
    await upload_ready_doc(client, headers, section_id, text="\n\n".join(parts), name="triangles.txt")

    await ask(client, headers, section_id, "i want to understand this chapter")

    messages = fake_llm.stream_calls[-1]
    prompt = messages[-1]["content"]
    assert "Part 0:" in prompt and "Part 59:" in prompt, "the chapter's start and end are both included"
    assert "teach it like a good tutor" in messages[0]["content"]


async def test_document_overview_question_is_answered_from_the_document(client, fake_llm):
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id, text=RESUME, name="resume.txt")

    events = parse_sse((await ask(client, headers, section_id, "what is this document about")).text)

    assert fake_llm.stream_calls, "the model should be asked to summarise the document"
    assert "Nitesh Raghuwanshi" in fake_llm.stream_calls[-1][-1]["content"]
    citations = next(data for name, data in events if name == "citations")["citations"]
    assert citations and citations[0]["filename"] == "resume.txt"
    assert citations[0]["matched_by"] == "overview"


async def test_off_topic_question_is_still_refused(client, fake_llm):
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id, text=RESUME, name="resume.txt")

    events = parse_sse((await ask(client, headers, section_id, "Who won the 2018 football world cup?")).text)

    assert "couldn't find" in "".join(data["text"] for name, data in events if name == "token")
    assert fake_llm.stream_calls == []
