"""Uploading the same file again reuses the existing document instead of processing a duplicate."""

import asyncio
import uuid
from pathlib import Path

from sqlalchemy import delete, func, select

from app.db import SessionLocal
from app.models import Document, DocumentHash, FigureScan
from app.services import documents, jobs

from .conftest import register
from .test_chat import DOC, new_section, upload_ready_doc


async def upload(client, headers, section_id, text=DOC, name="biology.txt"):
    response = await client.post(
        "/api/v1/documents",
        files={"file": (name, text.encode(), "text/plain")},
        data={"section_id": section_id},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    await asyncio.gather(*list(jobs._tasks), return_exceptions=True)
    return response.json()


async def document_count():
    async with SessionLocal() as session:
        return await session.scalar(select(func.count()).select_from(Document))


async def uploads_used(client, headers):
    return (await client.get("/api/v1/usage", headers=headers)).json()["limits"]["upload"]["used"]


async def test_same_file_in_the_same_chat(client):
    headers = await register(client)
    section_id = await new_section(client, headers)
    first = await upload_ready_doc(client, headers, section_id)

    again = await upload(client, headers, section_id)
    assert again["id"] == first and again["duplicate"] == "already_in_chat"
    assert await document_count() == 1
    assert await uploads_used(client, headers) == 1  # the repeat isn't counted
    docs = (await client.get(f"/api/v1/sections/{section_id}/documents", headers=headers)).json()
    assert [d["id"] for d in docs] == [first]


async def test_same_file_in_another_chat_is_attached_not_reprocessed(client, fake_llm):
    headers = await register(client)
    first_chat = await new_section(client, headers)
    doc_id = await upload_ready_doc(client, headers, first_chat)
    second_chat = await new_section(client, headers)

    again = await upload(client, headers, second_chat, name="renamed-copy.txt")  # same bytes, other name
    assert again["id"] == doc_id and again["duplicate"] == "existing" and again["status"] == "ready"
    assert await document_count() == 1
    docs = (await client.get(f"/api/v1/sections/{second_chat}/documents", headers=headers)).json()
    assert [d["id"] for d in docs] == [doc_id]

    # Different content is a new document, even with the same name.
    other = await upload(client, headers, second_chat, text="Mitochondria produce ATP for the cell.")
    assert other["id"] != doc_id and other["duplicate"] is None


async def test_other_students_files_are_never_reused(client):
    alice = await register(client, "alice@example.com")
    await upload_ready_doc(client, alice, await new_section(client, alice))
    bob = await register(client, "bob@example.com")
    bobs = await upload(client, bob, await new_section(client, bob))
    assert bobs["duplicate"] is None and await document_count() == 2


async def test_reupload_restores_a_lost_file_and_extracts_its_figures(client):
    """Uploaded before fingerprints and figure extraction existed, file since wiped by a restart:
    uploading it again reuses the document, keeps the new file and extracts the figures."""
    headers = await register(client)
    section_id = await new_section(client, headers)
    doc_id = uuid.UUID(await upload_ready_doc(client, headers, section_id))
    async with SessionLocal() as session:
        doc = await session.get(Document, doc_id)
        Path(doc.storage_path).unlink()
        await session.execute(delete(DocumentHash).where(DocumentHash.document_id == doc_id))
        await session.execute(delete(FigureScan).where(FigureScan.document_id == doc_id))
        await session.commit()
    await client.delete(f"/api/v1/sections/{section_id}/documents/{doc_id}", headers=headers)

    again = await upload(client, headers, section_id)
    assert again["id"] == str(doc_id) and again["duplicate"] == "existing"
    assert again["figure_status"] == "processing"  # extraction started
    async with SessionLocal() as session:
        doc = await session.get(Document, doc_id)
        assert Path(doc.storage_path).exists()
        assert (await session.get(FigureScan, doc_id)).status == "done"
        assert await session.get(DocumentHash, doc_id) is not None  # recognised by content next time
    assert await document_count() == 1


async def test_failed_documents_are_processed_again(client):
    headers = await register(client)
    section_id = await new_section(client, headers)
    doc_id = uuid.UUID(await upload_ready_doc(client, headers, section_id))
    async with SessionLocal() as session:
        (await session.get(Document, doc_id)).status = "failed"
        await session.commit()
    again = await upload(client, headers, section_id)
    assert again["id"] != str(doc_id) and again["duplicate"] is None


async def test_fingerprints_follow_documents(client):
    headers = await register(client)
    doc_id = uuid.UUID(await upload_ready_doc(client, headers, await new_section(client, headers)))
    async with SessionLocal() as session:
        await session.execute(delete(DocumentHash))
        await session.commit()
    await documents.backfill_hashes()  # earlier uploads whose file is still here
    async with SessionLocal() as session:
        assert await session.get(DocumentHash, doc_id) is not None

    await client.delete(f"/api/v1/documents/{doc_id}", headers=headers)
    async with SessionLocal() as session:
        assert await session.get(DocumentHash, doc_id) is None
