from app.services.rag import is_overview_question

from .conftest import parse_sse, register
from .test_chat import ask, new_section, upload_ready_doc

# Deliberately shares no words with "what is this document about".
RESUME = (
    "Nitesh Raghuwanshi. Software engineer skilled in Python, FastAPI and React.\n\n"
    "Experience: built retrieval augmented generation systems and data pipelines.\n\n"
    "Education: Bachelor of Technology in computer science."
)


def test_overview_detection():
    for question in ("what is this document about", "Summarize my notes", "give me the key points", "What is in the PDF?"):
        assert is_overview_question(question), question
    for question in ("what is photosynthesis", "Who won the 2018 football world cup?", "Explain enzymes"):
        assert not is_overview_question(question), question


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
