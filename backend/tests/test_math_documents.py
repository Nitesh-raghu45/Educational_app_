"""Math-heavy PDFs (e.g. NCERT textbooks): symbol fonts, LaTeX page reading, math-aware keyword
search and per-diagram figure extraction."""

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.models import DailyUsage, Document, DocumentChunk
from app.services import figures, ingestion, jobs, mathtext, providers
from app.services.keyword import tokenize

from .conftest import register
from .test_chat import new_section
from .test_figures import diagram, make_pdf

# What pypdf gives for an NCERT page: Symbol-font private-use characters, a fraction split over
# lines ("1" above "2") and powers written flat ("x2").
GARBLED_PAGE = (
    "Theorem 6.1 : In  ABC,  A   P and\n"
    "area  1\n2\n base  height, p(x)  x2  2x  8\n"
    "AD\nDB\n3\n2\n\n"
)
LATEX_PAGE = (
    "Theorem 6.1: In $\\triangle ABC$, $\\angle A = \\angle P$ and area $= \\frac{1}{2}$ base $\\times$ height, "
    "$p(x) = x^2 - 2x - 8$, $\\frac{AD}{DB} = \\frac{3}{2}$."
)


# ---------- symbol fonts ----------


def test_symbol_font_characters_become_real_symbols():
    assert mathtext.fix_symbol_font(" A   B,  PQR, 3  4  1, ") == (
        "∠ A = ∠ B, Δ PQR, 3 × 4 − 1, αβ"
    )
    assert mathtext.fix_symbol_font("plain text ∠ stays") == "plain text ∠ stays"


def test_pages_with_garbled_math_are_detected():
    assert mathtext.needs_math_reading(mathtext.fix_symbol_font(GARBLED_PAGE))
    assert not mathtext.needs_math_reading("Photosynthesis turns light into chemical energy.\n\n12\n")


def test_transcripts_that_lose_text_are_rejected():
    original = "Some page text " * 20
    assert mathtext._accept("```\n" + LATEX_PAGE + " " + original + "\n```", original).startswith("Theorem")
    assert mathtext._accept("", original) is None
    assert mathtext._accept("I can't read this image.", original) is None


# ---------- keyword search ----------


def test_keyword_tokens_match_across_math_notations():
    # A student types x², the PDF's text layer has x2 and the vision model writes x^{2}.
    assert "x2" in tokenize("zeroes of x²") and "x2" in tokenize("p(x) = x2 - 1") and "x2" in tokenize("$x^{2}$")
    # Section numbers stay whole (and keep their parts); LaTeX layout words are noise.
    assert tokenize("Theorem 6.1") == ["theorem", "6.1", "6", "1"]
    assert tokenize(r"\frac{AD}{DB} \left( \angle ABC \right)") == ["ad", "db", "angl", "abc"]


# ---------- page reading with the vision model ----------


class MathVision:
    def __init__(self, reply=LATEX_PAGE, fail=False):
        self.reply, self.fail, self.transcribed = reply, fail, 0

    async def transcribe(self, image_jpeg, prompt):
        self.transcribed += 1
        assert "LaTeX" in prompt
        if self.fail:
            raise RuntimeError("vision model down")
        return self.reply

    async def describe(self, image_jpeg, prompt):
        return json.dumps({"title": "Triangle ABC", "description": "A triangle with a line DE parallel to BC.", "useful": True})


@pytest.fixture
def math_pdf(tmp_path, monkeypatch):
    """A two-page PDF whose text layer looks like a garbled NCERT page."""
    path = make_pdf(tmp_path / "triangles.pdf", [[(diagram(seed=1), (100, 300, 400, 300))], []])
    monkeypatch.setattr(
        ingestion, "extract_text", lambda p: ([(1, mathtext.fix_symbol_font(GARBLED_PAGE)), (2, "Plain words only here.")], 0)
    )
    monkeypatch.setattr(mathtext, "render_pages", lambda p, numbers: {n: b"\xff\xd8jpeg" for n in numbers})
    return path


def use_vision(fake_llm, vision):
    providers.override(embedder=providers.embedder(), store=providers.vector_store(), llm=fake_llm, vision=vision)


async def upload_pdf(client, headers, path) -> uuid.UUID:
    response = await client.post(
        "/api/v1/documents",
        files={"file": (path.name, path.read_bytes(), "application/pdf")},
        data={"section_id": await new_section(client, headers)},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    await asyncio.gather(*list(jobs._tasks), return_exceptions=True)
    return uuid.UUID(response.json()["id"])


async def chunk_texts(doc_id) -> tuple[str, list[str]]:
    async with SessionLocal() as session:
        status = (await session.get(Document, doc_id)).status
        texts = list(await session.scalars(select(DocumentChunk.text).where(DocumentChunk.document_id == doc_id)))
    return status, texts


async def math_pages_used() -> int:
    async with SessionLocal() as session:
        return sum(await session.scalars(select(DailyUsage.count).where(DailyUsage.kind == "math_page")))


async def test_math_pages_are_read_as_latex(client, fake_llm, math_pdf):
    vision = MathVision()
    use_vision(fake_llm, vision)
    doc_id = await upload_pdf(client, await register(client), math_pdf)
    status, texts = await chunk_texts(doc_id)
    joined = " ".join(texts)
    assert status == "ready" and vision.transcribed == 1  # only the math page
    assert r"\frac{AD}{DB} = \frac{3}{2}" in joined and "Plain words only here." in joined
    assert "" not in joined and await math_pages_used() == 1


async def test_without_the_vision_model_pages_keep_their_fixed_text(client, fake_llm, math_pdf):
    vision = MathVision(fail=True)
    use_vision(fake_llm, vision)
    doc_id = await upload_pdf(client, await register(client), math_pdf)
    status, texts = await chunk_texts(doc_id)
    joined = " ".join(texts)
    assert status == "ready" and vision.transcribed == 1
    assert "∠ A = ∠ P" in joined and r"\frac" not in joined
    assert await math_pages_used() == 0  # the failed reading was refunded


async def test_math_page_reading_respects_the_daily_limit(client, fake_llm, math_pdf, monkeypatch):
    monkeypatch.setattr(get_settings(), "daily_math_page_limit", 1)
    vision = MathVision()
    use_vision(fake_llm, vision)
    headers = await register(client)
    await upload_pdf(client, headers, math_pdf)
    other = make_pdf(math_pdf.with_name("similarity.pdf"), [[(diagram(seed=2), (100, 300, 400, 300))], []])
    doc_id = await upload_pdf(client, headers, other)  # different bytes, so not a duplicate
    status, texts = await chunk_texts(doc_id)
    assert status == "ready" and vision.transcribed == 1  # the second upload is over today's limit
    assert "∠ A = ∠ P" in " ".join(texts)


async def test_math_page_reading_can_be_turned_off(client, fake_llm, math_pdf, monkeypatch):
    monkeypatch.setattr(get_settings(), "math_vision_enabled", False)
    vision = MathVision()
    use_vision(fake_llm, vision)
    status, _ = await chunk_texts(await upload_pdf(client, await register(client), math_pdf))
    assert status == "ready" and vision.transcribed == 0


# ---------- vector diagrams ----------


def path(left, bottom, right, top):
    return SimpleNamespace(get_pos=lambda: (left, bottom, right, top))


def test_each_vector_diagram_is_found_on_its_own():
    visible = (18, 18, 486, 702)  # inset crop box, like NCERT pages
    triangle = [path(100 + i * 10, 400, 125 + i * 10, 480) for i in range(8)]  # touching shapes
    graph = [path(300, 150 + i * 8, 440, 156 + i * 8) for i in range(20)]
    objects = (
        triangle
        + graph
        + [path(20, 660, 484, 661)]  # header rule passing near nothing
        + [path(150 + i * 40, 300, 170 + i * 40, 300.5) for i in range(3)]  # fraction bars
        + [path(0, 0, 10, 10)]  # printer's crop mark outside the visible page
        + [path(18, 18, 486, 702)]  # page frame
    )
    page = SimpleNamespace(get_objects=lambda filter: objects)
    regions = figures._vector_regions(page, SimpleNamespace(FPDF_PAGEOBJ_PATH=2), visible, image_boxes=[])
    m = figures.VECTOR_MARGIN_PT
    assert regions == [(100 - m, 400 - m, 195 + m, 480 + m), (300 - m, 150 - m, 440 + m, 308 + m)]
    # A cluster around a photo is its frame, not a separate diagram.
    assert len(figures._vector_regions(page, SimpleNamespace(FPDF_PAGEOBJ_PATH=2), visible, [(90, 390, 200, 490)])) == 1


def test_touching_boxes_cluster_together():
    clusters = figures.cluster_boxes([(0, 0, 10, 10), (12, 0, 20, 10), (100, 100, 110, 110)], gap=3)
    assert sorted(clusters) == [(1, (100, 100, 110, 110)), (2, (0, 0, 20, 10))]


def test_boxes_further_apart_than_the_gap_stay_separate():
    clusters = figures.cluster_boxes([(0, 0, 10, 10), (18, 0, 30, 10)], gap=6)
    assert len(clusters) == 2


def test_a_page_drawn_as_tens_of_thousands_of_shapes_clusters_in_seconds():
    # NCERT science pages draw photos as 50,000-70,000 tiny vector shapes; comparing every
    # pair took hours and left the figure scan "processing" forever.
    import time

    illustration = [(100 + (i % 250) * 1.2, 300 + (i // 250) * 1.2, 101 + (i % 250) * 1.2, 301 + (i // 250) * 1.2) for i in range(50_000)]
    began = time.perf_counter()
    clusters = figures.cluster_boxes(illustration + [(500, 50, 540, 90)], gap=6)
    assert time.perf_counter() - began < 10
    assert sorted(count for count, _ in clusters) == [1, 50_000]
