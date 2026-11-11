"""Uploaded files are kept in the database too, so a restart that wipes the disk loses nothing."""

import asyncio
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.config import get_settings
from app.db import SessionLocal
from app.main import backfill_figures
from app.models import Document, DocumentFilePart, FigureImage, FigureScan
from app.services import ingestion, jobs, stored_files

from .conftest import register
from .test_figures import diagram, new_section, upload_image, vision  # noqa: F401  (vision is a fixture)


@pytest.fixture(autouse=True)
def keep_copies(monkeypatch):
    """These tests restore files mid-processing, while a copy still exists; normally it is
    released once processing finishes (tested at the end of this file)."""
    monkeypatch.setattr(get_settings(), "keep_file_copies", True)


async def wipe_disk(doc_id) -> Document:
    async with SessionLocal() as session:
        doc = await session.get(Document, doc_id)
    Path(doc.storage_path).unlink()
    return doc


async def test_restart_during_figure_extraction_finishes_after_it(client, vision, monkeypatch):
    headers = await register(client)
    doc_id = await upload_image(client, headers, await new_section(client, headers), monkeypatch)
    original = Path((await wipe_disk(doc_id)).storage_path)
    async with SessionLocal() as session:
        (await session.get(FigureScan, doc_id)).status = "processing"  # cut off by the restart
        await session.commit()

    await backfill_figures()  # runs at startup
    await asyncio.gather(*list(jobs._tasks), return_exceptions=True)
    async with SessionLocal() as session:
        scan = await session.get(FigureScan, doc_id)
        assert (scan.status, scan.figure_count) == ("done", 1)
        assert await session.scalar(select(func.count()).select_from(FigureImage).where(FigureImage.document_id == doc_id)) == 1
    assert original.exists()  # restored from the database


async def test_processing_again_restores_the_file(client, vision, monkeypatch):
    headers = await register(client)
    doc_id = await upload_image(client, headers, await new_section(client, headers), monkeypatch)
    await wipe_disk(doc_id)
    await ingestion.ingest_document(doc_id)
    async with SessionLocal() as session:
        assert (await session.get(Document, doc_id)).status == "ready"


async def test_large_files_are_split_and_restored_exactly(client, vision, monkeypatch):
    monkeypatch.setattr(stored_files, "PART_BYTES", 1000)
    headers = await register(client)
    doc_id = await upload_image(client, headers, await new_section(client, headers), monkeypatch, image=diagram(seed=3))
    doc = await wipe_disk(doc_id)
    async with SessionLocal() as session:
        parts = await session.scalar(select(func.count()).select_from(DocumentFilePart).where(DocumentFilePart.document_id == doc_id))
        assert parts == -(-doc.size_bytes // 1000)
        assert await stored_files.ensure_local(session, doc)
    assert Path(doc.storage_path).stat().st_size == doc.size_bytes

    await client.delete(f"/api/v1/documents/{doc_id}", headers=headers)
    async with SessionLocal() as session:
        assert not await stored_files.has_copy(session, doc_id)


async def test_copy_is_released_once_fully_processed(client, vision, monkeypatch):
    monkeypatch.setattr(get_settings(), "keep_file_copies", False)
    headers = await register(client)
    section_id = await new_section(client, headers)
    doc_id = await upload_image(client, headers, section_id, monkeypatch)
    async with SessionLocal() as session:
        assert (await session.get(FigureScan, doc_id)).status == "done"
        assert not await stored_files.has_copy(session, doc_id)  # text and figures done: not needed

    # Uploading the same file again needs nothing processed, so no copy comes back...
    await upload_image(client, headers, section_id, monkeypatch)
    async with SessionLocal() as session:
        assert not await stored_files.has_copy(session, doc_id)
        (await session.get(FigureScan, doc_id)).status = "failed"
        await session.commit()
    # ...unless its figures still have to be extracted: then it's kept until that's done.
    await upload_image(client, headers, section_id, monkeypatch)
    async with SessionLocal() as session:
        assert (await session.get(FigureScan, doc_id)).status == "done"
        assert not await stored_files.has_copy(session, doc_id)
