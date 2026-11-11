"""The shared library: admin-uploaded books that every user can add to their chats and ask about,
while everyone's own uploads stay private."""

import asyncio
import uuid

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.models import DocumentFigure, LibraryDocument
from app.services import ingestion, jobs, library, providers

from .conftest import parse_sse, register
from .test_chat import DOC, ask, new_section, upload_ready_doc
from .test_figures import diagram, png_bytes, vision  # noqa: F401  (vision is a fixture)

ADMIN = "admin@example.com"
PHYSICS = "Newton's second law: force equals mass times acceleration. The SI unit of force is the newton."


@pytest.fixture(autouse=True)
def admin_email(monkeypatch):
    monkeypatch.setattr(get_settings(), "admin_emails", f"Teacher@Example.com, {ADMIN}")
    monkeypatch.setattr(library, "_admin_cache", None)


async def add_book(client, headers, text=PHYSICS, name="Class 9 Ch 9 Force.txt", grade="9", subject="Science"):
    response = await client.post(
        "/api/v1/library",
        files={"file": (name, text.encode(), "text/plain")},
        data={"grade": grade, "subject": subject},
        headers=headers,
    )
    await asyncio.gather(*list(jobs._tasks), return_exceptions=True)
    return response


async def test_admins_add_books_that_every_user_can_ask_about(client):
    admin = await register(client, ADMIN)
    assert (await client.get("/api/v1/auth/me", headers=admin)).json()["is_admin"] is True
    book = await add_book(client, admin)
    assert book.status_code == 201, book.text
    book_id = book.json()["id"]

    student = await register(client, "student@example.com")
    assert (await client.get("/api/v1/auth/me", headers=student)).json()["is_admin"] is False
    shelf = (await client.get("/api/v1/library", headers=student)).json()
    assert [(d["id"], d["grade"], d["subject"], d["library"], d["status"]) for d in shelf] == [
        (book_id, "9", "Science", True, "ready")
    ]

    section_id = await new_section(client, student)
    attached = await client.post(f"/api/v1/sections/{section_id}/documents", json={"document_id": book_id}, headers=student)
    assert attached.status_code == 201 and attached.json()["library"] is True
    events = parse_sse((await ask(client, student, section_id, "What is the SI unit of force?")).text)
    citations = next(d for n, d in events if n == "citations")["citations"]
    assert citations and all(c["document_id"] == book_id for c in citations)

    # The book's vectors are the library's, not the admin's, so any user's search matches them.
    vectors = providers.vector_store().namespaces[providers.namespace_for()]
    assert {meta["user_id"] for _, meta in vectors.values()} == {library.LIBRARY_OWNER}
    # It's listed by the library, not among the admin's own documents.
    assert (await client.get("/api/v1/documents", headers=admin)).json() == []


async def test_only_admins_change_the_library(client):
    admin = await register(client, ADMIN)
    book_id = (await add_book(client, admin)).json()["id"]
    student = await register(client, "student@example.com")

    assert (await add_book(client, student)).status_code == 403
    assert (await client.delete(f"/api/v1/library/{book_id}", headers=student)).status_code == 403
    assert (await client.delete(f"/api/v1/documents/{book_id}", headers=student)).status_code == 404

    section_id = await new_section(client, student)
    await client.post(f"/api/v1/sections/{section_id}/documents", json={"document_id": book_id}, headers=student)
    assert (await client.delete(f"/api/v1/library/{book_id}", headers=admin)).status_code == 204
    assert (await client.get(f"/api/v1/sections/{section_id}/documents", headers=student)).json() == []
    assert (await client.get("/api/v1/library", headers=student)).json() == []


async def test_picking_a_subject_swaps_the_chats_library_chapters(client):
    admin = await register(client, ADMIN)
    force = (await add_book(client, admin)).json()["id"]
    motion = (await add_book(client, admin, "Speed is distance travelled per unit time.", "Ch 8 Motion.txt")).json()["id"]
    algebra = (await add_book(client, admin, "A polynomial of degree two is quadratic.", "Ch 2 Polynomials.txt", "10", "Mathematics")).json()["id"]

    student = await register(client, "student@example.com")
    section_id = await new_section(client, student)
    own = await upload_ready_doc(client, student, section_id)
    url = f"/api/v1/sections/{section_id}/library"

    # Class 9 Science, all chapters.
    docs = (await client.put(url, json={"document_ids": [force, motion]}, headers=student)).json()
    assert {d["id"] for d in docs} == {own, force, motion}
    # Switched to Class 10 Mathematics: the science chapters go, the student's own file stays.
    docs = (await client.put(url, json={"document_ids": [algebra]}, headers=student)).json()
    assert {d["id"] for d in docs} == {own, algebra}
    # A student's own document can't be passed off as a library chapter.
    assert (await client.put(url, json={"document_ids": [own]}, headers=student)).status_code == 404
    # Nothing picked: only their own file is left.
    docs = (await client.put(url, json={"document_ids": []}, headers=student)).json()
    assert [d["id"] for d in docs] == [own]
    # Someone else's chat can't be changed.
    other = await register(client, "other@example.com")
    assert (await client.put(url, json={"document_ids": [force]}, headers=other)).status_code == 404


async def test_own_uploads_stay_private(client):
    alice, bob = await register(client, "alice@example.com"), await register(client, "bob@example.com")
    alice_doc = await upload_ready_doc(client, alice, await new_section(client, alice))
    bob_section = await new_section(client, bob)
    attach = await client.post(f"/api/v1/sections/{bob_section}/documents", json={"document_id": alice_doc}, headers=bob)
    assert attach.status_code == 404
    assert (await client.get("/api/v1/library", headers=bob)).json() == []


async def test_users_see_only_ready_books_and_admins_see_all(client, monkeypatch):
    admin = await register(client, ADMIN)

    async def stuck(document_id):  # still being processed
        pass

    monkeypatch.setattr(ingestion, "ingest_document", stuck)
    await add_book(client, admin)
    student = await register(client, "student@example.com")
    assert (await client.get("/api/v1/library", headers=student)).json() == []
    assert [d["status"] for d in (await client.get("/api/v1/library", headers=admin)).json()] == ["pending"]


async def test_admins_have_no_daily_limits(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "daily_upload_limit", 1)
    admin = await register(client, "teacher@example.com")  # matched case-insensitively
    for n in range(3):
        response = await add_book(client, admin, text=f"{PHYSICS} Part {n}.", name=f"part-{n}.txt")
        assert response.status_code == 201, response.text
    student = await register(client, "student@example.com")
    section_id = await new_section(client, student)
    await upload_ready_doc(client, student, section_id)
    second = await client.post(
        "/api/v1/documents", files={"file": ("b.txt", b"Other notes.", "text/plain")},
        data={"section_id": section_id}, headers=student,
    )
    assert second.status_code == 429  # students keep theirs


async def test_sharing_an_own_document_reindexes_it_for_everyone(client):
    admin = await register(client, ADMIN)
    own = await upload_ready_doc(client, admin, await new_section(client, admin))
    store = providers.vector_store().namespaces[providers.namespace_for()]
    assert {m["user_id"] for _, m in store.values()} != {library.LIBRARY_OWNER}

    shared = await add_book(client, admin, text=DOC, name="biology.txt", grade="10", subject="Biology")
    assert shared.json()["id"] == own and shared.json()["library"] is True
    assert {m["user_id"] for _, m in store.values()} == {library.LIBRARY_OWNER}
    async with SessionLocal() as session:
        entry = await session.get(LibraryDocument, uuid.UUID(own))
        assert (entry.grade, entry.subject) == ("10", "Biology")


async def test_library_figures_are_visible_to_every_user(client, vision, monkeypatch):
    monkeypatch.setattr(ingestion, "ocr_image", lambda image: "The water cycle: evaporation, condensation, precipitation.")
    admin = await register(client, ADMIN)
    response = await client.post(
        "/api/v1/library", files={"file": ("water-cycle.png", png_bytes(diagram()), "image/png")},
        data={"grade": "7", "subject": "Science"}, headers=admin,
    )
    await asyncio.gather(*list(jobs._tasks), return_exceptions=True)
    async with SessionLocal() as session:
        figure = await session.scalar(
            select(DocumentFigure).where(DocumentFigure.document_id == uuid.UUID(response.json()["id"]))
        )
    student = await register(client, "student@example.com")
    image = await client.get(f"/api/v1/figures/{figure.id}", headers=student)
    assert image.status_code == 200 and image.headers["content-type"] == "image/webp"
