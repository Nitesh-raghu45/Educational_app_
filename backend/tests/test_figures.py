"""Figures from documents: extraction, description, search, answers, access control, cleanup."""

import asyncio
import io
import json
import uuid

import pytest
from PIL import Image, ImageDraw
from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.figure_store import figure_store
from app.models import DocumentChunk, DocumentFigure
from app.services import figures, ingestion, jobs, providers
from app.services.rag import extract_citations
from app.services.retrieval import Evidence, format_sources

from .conftest import parse_sse, register


class FakeVision:
    def __init__(self, reply=None):
        self.calls = 0
        self.reply = reply or {
            "title": "Water cycle diagram",
            "description": "Arrows show evaporation from the ocean, condensation into clouds and precipitation as rain.",
            "useful": True,
        }

    async def describe(self, image_jpeg, prompt):
        self.calls += 1
        assert image_jpeg[:2] == b"\xff\xd8"  # the model is sent a JPEG
        return json.dumps(self.reply)


@pytest.fixture
def vision(fake_llm):
    fake = FakeVision()
    providers.override(
        embedder=providers.embedder(), store=providers.vector_store(), llm=fake_llm, vision=fake
    )
    return fake


def diagram(width=400, height=300, seed=0) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for i in range(4):
        box = [min(20 + i * 10 + seed, width - 2), min(20 + i * 60, height - 2), width - 1, height - 1]
        draw.rectangle([box[0], box[1], max(box[0], width - 40), max(box[1], min(60 + i * 60, height - 1))], outline="black", width=3)
    draw.line([0, seed * 7 % height, width, height - 1], fill="blue", width=4)
    return image


def png_bytes(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


async def upload_image(client, headers, section_id, monkeypatch, name="water-cycle.png") -> str:
    monkeypatch.setattr(ingestion, "ocr_image", lambda image: "The water cycle: evaporation, condensation, precipitation.")
    response = await client.post(
        "/api/v1/documents",
        files={"file": (name, png_bytes(diagram()), "image/png")},
        data={"section_id": section_id},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    await asyncio.gather(*list(jobs._tasks), return_exceptions=True)
    return uuid.UUID(response.json()["id"])


async def new_section(client, headers) -> str:
    return (await client.post("/api/v1/sections", json={}, headers=headers)).json()["id"]


def make_pdf(path, pages):
    """pages: list of lists of (PIL image, (x, y, width, height) in points)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    for objects in pages:
        page = pdf.new_page(612, 792)
        for image, (x, y, w, h) in objects:
            obj = pdfium.PdfImage.new(pdf)
            obj.set_bitmap(pdfium.PdfBitmap.from_pil(image))
            obj.set_matrix(pdfium.PdfMatrix().scale(w, h).translate(x, y))
            page.insert_obj(obj)
        page.gen_content()
        page.close()
    pdf.save(str(path))
    pdf.close()
    return path


def test_pdf_figures_skip_logos_icons_and_lines(tmp_path):
    logo = diagram(200, 200, seed=5)
    pdf = make_pdf(
        tmp_path / "notes.pdf",
        [
            [(logo, (500, 700, 80, 80))],
            [(logo, (500, 700, 80, 80)), (diagram(seed=1), (100, 300, 400, 300)), (diagram(60, 60), (50, 50, 30, 30))],
            [(logo, (500, 700, 80, 80)), (diagram(900, 40, seed=2), (20, 100, 560, 25))],
        ],
    )
    found = figures.extract_candidates(pdf, limit=30)
    # The logo repeats on every page, the icon is tiny and the banner is a thin strip.
    assert [c.page for c in found] == [2]
    assert found[0].webp[:4] == b"RIFF" and max(found[0].width, found[0].height) <= figures.STORE_MAX_SIDE


def test_captions_must_be_useful_json():
    assert figures._parse_caption("not json") is None
    assert figures._parse_caption(json.dumps({"title": "Logo", "description": "A logo", "useful": False})) is None
    assert figures._parse_caption(json.dumps({"title": "", "description": "x"})) is None
    assert figures._parse_caption(json.dumps({"title": " Cell  diagram ", "description": "Labels: nucleus."})) == (
        "Cell diagram",
        "Labels: nucleus.",
    )


def test_figure_sources_are_marked_and_figure_references_are_cited():
    plain = Evidence("d-0", None, "notes.pdf", 1, "Evaporation happens.", 0.9)
    figure = Evidence(
        "d-fig-0", None, "notes.pdf", 2, "Figure (page 2): Water cycle.", 0.8,
        figure={"id": "f1", "title": "Water cycle", "width": 400, "height": 300},
    )
    assert 'kind="figure"' in format_sources([plain, figure]) and format_sources([plain]).count("kind=") == 0
    cited = extract_citations("Water evaporates [1].\n\n[[figure:2]]\n\n[[figure:1]]", [plain, figure])
    # [[figure:1]] points at a text source, so it is ignored.
    assert [c["index"] for c in cited] == [1, 2]
    assert cited[1]["figure"]["id"] == "f1" and "figure" not in cited[0]


async def test_figures_are_found_described_searched_and_shown(client, vision, monkeypatch):
    headers = await register(client)
    section_id = await new_section(client, headers)
    doc_id = await upload_image(client, headers, section_id, monkeypatch)

    docs = (await client.get(f"/api/v1/sections/{section_id}/documents", headers=headers)).json()
    assert docs[0]["status"] == "ready"
    assert (docs[0]["figure_status"], docs[0]["figure_count"]) == ("done", 1)
    assert vision.calls == 1

    async with SessionLocal() as session:
        figure = await session.scalar(select(DocumentFigure).where(DocumentFigure.document_id == doc_id))
        chunk = await session.get(DocumentChunk, figure.chunk_id)
    assert chunk.chunk_index >= figures.FIGURE_CHUNK_BASE and "Water cycle diagram" in chunk.text

    async def stream(messages, model, **kwargs):
        sources = messages[-1]["content"]
        n = next(i for i in range(1, 13) if f'<source id="{i}"' in sources and 'kind="figure"' in sources.split(f'<source id="{i}"')[1].split(">")[0])
        for piece in ["Water evaporates and condenses [1].\n\n", f"[[figure:{n}]]\n\nThen it rains."]:
            yield piece

    monkeypatch.setattr(providers.llm(), "stream", stream)
    response = await client.post(
        f"/api/v1/sections/{section_id}/messages",
        json={"content": "Show the water cycle diagram with evaporation and condensation", "client_message_id": uuid.uuid4().hex},
        headers=headers,
    )
    citations = next(data for name, data in parse_sse(response.text) if name == "citations")["citations"]
    shown = [c for c in citations if c.get("figure")]
    assert len(shown) == 1 and shown[0]["figure"]["id"] == str(figure.id)

    image = await client.get(f"/api/v1/figures/{figure.id}", headers=headers)
    assert image.status_code == 200 and image.headers["content-type"] == "image/webp"
    assert Image.open(io.BytesIO(image.content)).size == (figure.width, figure.height)

    # Another student can't load it, and nobody can without signing in.
    other = await register(client, "other@example.com")
    assert (await client.get(f"/api/v1/figures/{figure.id}", headers=other)).status_code == 404
    assert (await client.get(f"/api/v1/figures/{figure.id}")).status_code == 401


async def test_deleting_a_document_deletes_its_figures(client, vision, monkeypatch):
    headers = await register(client)
    doc_id = await upload_image(client, headers, await new_section(client, headers), monkeypatch)
    async with SessionLocal() as session:
        figure = await session.scalar(select(DocumentFigure).where(DocumentFigure.document_id == doc_id))
    assert figure_store().get(figure.storage_key) is not None

    assert (await client.delete(f"/api/v1/documents/{doc_id}", headers=headers)).status_code == 204
    assert figure_store().get(figure.storage_key) is None
    async with SessionLocal() as session:
        assert await session.get(DocumentFigure, figure.id) is None
        assert await session.get(DocumentChunk, figure.chunk_id) is None


async def test_useless_images_and_the_daily_limit_add_no_figures(client, vision, monkeypatch):
    headers = await register(client)
    vision.reply = {"title": "Logo", "description": "A company logo.", "useful": False}
    doc_id = await upload_image(client, headers, await new_section(client, headers), monkeypatch)
    docs = (await client.get("/api/v1/documents", headers=headers)).json()
    assert (docs[0]["figure_status"], docs[0]["figure_count"]) == ("done", 0)

    monkeypatch.setattr(get_settings(), "daily_figure_limit", 1)  # 1 already used above
    vision.reply = FakeVision().reply
    calls = vision.calls
    await upload_image(client, headers, await new_section(client, headers), monkeypatch, name="again.png")
    assert vision.calls == calls  # limit reached: the vision model isn't called
    docs = (await client.get("/api/v1/documents", headers=headers)).json()
    assert docs[0]["status"] == "ready" and docs[0]["figure_count"] == 0
    assert str(doc_id) != docs[0]["id"]


async def test_figure_failures_never_break_the_document(client, monkeypatch, fake_llm):
    class BrokenVision:
        async def describe(self, image_jpeg, prompt):
            raise RuntimeError("vision model down")

    providers.override(embedder=providers.embedder(), store=providers.vector_store(), llm=fake_llm, vision=BrokenVision())
    headers = await register(client)
    await upload_image(client, headers, await new_section(client, headers), monkeypatch)
    docs = (await client.get("/api/v1/documents", headers=headers)).json()
    assert docs[0]["status"] == "ready" and docs[0]["figure_count"] == 0 and docs[0]["figure_status"] == "done"


def test_r2_store_uses_the_bucket_correctly():
    from botocore.stub import Stubber

    from app.config import Settings
    from app.figure_store import R2FigureStore

    store = R2FigureStore(
        Settings(r2_account_id="acc123", r2_access_key_id="key", r2_secret_access_key="secret", r2_bucket="figs")
    )
    assert store.client.meta.endpoint_url == "https://acc123.r2.cloudflarestorage.com"
    with Stubber(store.client) as stub:
        stub.add_response(
            "put_object", {}, {"Bucket": "figs", "Key": "a.webp", "Body": b"img", "ContentType": "image/webp"}
        )
        stub.add_client_error("get_object", "NoSuchKey", http_status_code=404)
        stub.add_response(
            "delete_objects",
            {},
            {"Bucket": "figs", "Delete": {"Objects": [{"Key": "a.webp"}], "Quiet": True}},
        )
        store.put("a.webp", b"img", "image/webp")
        assert store.get("missing.webp") is None
        store.delete(["a.webp"])
        stub.assert_no_pending_responses()
