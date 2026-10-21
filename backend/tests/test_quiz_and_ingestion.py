import uuid

from app.services.ingestion import split_text

from .conftest import register
from .test_chat import new_section, upload_ready_doc


def test_split_text_overlaps_and_respects_size():
    text = " ".join(f"word{i}." for i in range(600))
    chunks = split_text(text, size=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    # Consecutive chunks share some text.
    assert chunks[0][-40:].split()[-1] in chunks[1]


def test_split_text_empty():
    assert split_text("   \n\n  ", 500, 100) == []


async def test_unsupported_upload_rejected(client):
    headers = await register(client)
    response = await client.post(
        "/api/v1/documents", files={"file": ("virus.exe", b"MZ", "application/octet-stream")}, headers=headers
    )
    assert response.status_code == 415


async def test_quiz_generation_and_scoring(client):
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)

    response = await client.post(f"/api/v1/sections/{section_id}/quizzes", json={"num_questions": 3}, headers=headers)
    assert response.status_code == 201, response.text
    quiz = response.json()
    # The invalid question (unknown source) was dropped; answers are not exposed.
    assert len(quiz["questions"]) == 1
    assert "answer_index" not in quiz["questions"][0]

    attempt = await client.post(f"/api/v1/quizzes/{quiz['id']}/attempts", json={"answers": [0]}, headers=headers)
    assert attempt.status_code == 201
    body = attempt.json()
    assert (body["score"], body["total"]) == (1, 1)
    assert body["results"][0]["citation"]["filename"] == "biology.txt"


async def test_detached_document_is_not_searched(client):
    headers = await register(client)
    section_id = await new_section(client, headers)
    doc_id = await upload_ready_doc(client, headers, section_id)
    await client.delete(f"/api/v1/sections/{section_id}/documents/{doc_id}", headers=headers)
    response = await client.post(
        f"/api/v1/sections/{section_id}/messages", json={"content": "What is photosynthesis?"}, headers=headers
    )
    assert response.json()["error"]["code"] == "NO_READY_DOCUMENTS"


def test_citations_accept_gpt_oss_bracket_styles():
    from app.services.rag import extract_citations
    from app.services.retrieval import Evidence

    evidence = [Evidence(f"c{i}", uuid.uuid4(), f"doc{i}.pdf", i, "text", 0.9) for i in (1, 2, 3)]
    found = extract_citations("A [1]. B 【2】. C 【3†source】. D [9].", evidence)
    assert [c["index"] for c in found] == [1, 2, 3]


def test_recursive_chunking_keeps_paragraphs_whole():
    paragraphs = [f"Paragraph {i}. " + "This sentence explains one idea in detail. " * 6 for i in range(6)]
    chunks = split_text("\n\n".join(paragraphs), size=1000, overlap=150)
    assert all(len(c) <= 1000 for c in chunks)
    # Every paragraph (~270 chars) appears unbroken inside some chunk.
    for paragraph in paragraphs:
        assert any(paragraph.strip() in chunk for chunk in chunks), paragraph[:20]


def test_recursive_chunking_splits_long_paragraph_at_sentences():
    sentence = "Mitochondria produce ATP through cellular respiration. "
    chunks = split_text(sentence * 60, size=500, overlap=100)
    assert len(chunks) > 1 and all(len(c) <= 500 for c in chunks)
    # Chunks end on a sentence boundary rather than mid-word.
    assert all(c.endswith(".") for c in chunks)
    # Neighbouring chunks overlap by whole sentences.
    assert chunks[1].startswith("Mitochondria")


def test_recursive_chunking_handles_text_without_separators():
    chunks = split_text("x" * 2500, size=1000, overlap=150)
    assert [len(c) for c in chunks] == [1000, 1000, 500]


async def test_restart_resumes_unfinished_documents_and_reports_lost_files(client):
    import uuid as _uuid

    from app.db import SessionLocal
    from app.main import resume_unfinished_ingestion
    from app.models import Document

    headers = await register(client)
    user_id = _uuid.UUID((await client.get("/api/v1/auth/me", headers=headers)).json()["id"])
    async with SessionLocal() as session:
        doc = Document(
            user_id=user_id, filename="lost.txt", content_type="text/plain", size_bytes=10,
            storage_path="/nonexistent/lost.txt", status="processing",
        )
        session.add(doc)
        await session.commit()
        doc_id = doc.id

    await resume_unfinished_ingestion()
    import asyncio

    from app.services import jobs
    await asyncio.gather(*list(jobs._tasks), return_exceptions=True)

    status = (await client.get(f"/api/v1/documents/{doc_id}/status", headers=headers)).json()
    assert status["status"] == "failed"
    assert "upload it again" in status["error"]


async def test_ingestion_logs_time_per_step(client, caplog):
    import logging

    caplog.set_level(logging.INFO)
    headers = await register(client)
    section_id = await new_section(client, headers)
    await upload_ready_doc(client, headers, section_id)
    upload_line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Upload "))
    ingest_line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Ingested document"))
    assert "'biology.txt'" in upload_line and "saved in" in upload_line
    for part in ("'biology.txt'", "chunks", "parse ", "chunk ", "embed ", "pinecone ", "database ", "total "):
        assert part in ingest_line, (part, ingest_line)
    print("\n" + upload_line + "\n" + ingest_line)
