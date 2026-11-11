"""Readable text from math-heavy PDFs, such as NCERT textbooks.

1. Symbol-font characters. Many PDFs draw ∠ Δ = − × α … with the classic "Symbol" font, and
   their text layer gives those as private-use characters (U+F020-U+F0FF) that neither the answer
   model nor keyword search can read. `fix_symbol_font` maps them back to real Unicode.
2. Layout. A PDF's text layer loses the layout of fractions, powers and roots ("x2" for x², a
   fraction split over two lines). Pages that look like that are read again from a picture of
   the page by the vision model, which writes the math in LaTeX (`read_math_pages`).
"""

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from ..config import get_settings
from ..db import SessionLocal
from ..errors import AppError
from ..models import PageTranscript
from . import providers, usage
from .vision import is_retryable

logger = logging.getLogger(__name__)

# Adobe Symbol encoding, codes 0x20-0xFE: text layers report code C as U+F000 + C.
# Big-bracket pieces: the top piece becomes the bracket, the middle and bottom pieces vanish.
_SYMBOL_FONT = {
    0x20: " ", 0x21: "!", 0x22: "∀", 0x23: "#", 0x24: "∃", 0x25: "%", 0x26: "&", 0x27: "∋",
    0x28: "(", 0x29: ")", 0x2A: "∗", 0x2B: "+", 0x2C: ",", 0x2D: "−", 0x2E: ".", 0x2F: "/",
    **{0x30 + d: str(d) for d in range(10)},
    0x3A: ":", 0x3B: ";", 0x3C: "<", 0x3D: "=", 0x3E: ">", 0x3F: "?", 0x40: "≅",
    0x41: "Α", 0x42: "Β", 0x43: "Χ", 0x44: "Δ", 0x45: "Ε", 0x46: "Φ", 0x47: "Γ", 0x48: "Η",
    0x49: "Ι", 0x4A: "ϑ", 0x4B: "Κ", 0x4C: "Λ", 0x4D: "Μ", 0x4E: "Ν", 0x4F: "Ο", 0x50: "Π",
    0x51: "Θ", 0x52: "Ρ", 0x53: "Σ", 0x54: "Τ", 0x55: "Υ", 0x56: "ς", 0x57: "Ω", 0x58: "Ξ",
    0x59: "Ψ", 0x5A: "Ζ", 0x5B: "[", 0x5C: "∴", 0x5D: "]", 0x5E: "⊥", 0x5F: "_", 0x60: "",
    0x61: "α", 0x62: "β", 0x63: "χ", 0x64: "δ", 0x65: "ε", 0x66: "φ", 0x67: "γ", 0x68: "η",
    0x69: "ι", 0x6A: "ϕ", 0x6B: "κ", 0x6C: "λ", 0x6D: "μ", 0x6E: "ν", 0x6F: "ο", 0x70: "π",
    0x71: "θ", 0x72: "ρ", 0x73: "σ", 0x74: "τ", 0x75: "υ", 0x76: "ϖ", 0x77: "ω", 0x78: "ξ",
    0x79: "ψ", 0x7A: "ζ", 0x7B: "{", 0x7C: "|", 0x7D: "}", 0x7E: "∼",
    0xA1: "ϒ", 0xA2: "′", 0xA3: "≤", 0xA4: "⁄", 0xA5: "∞", 0xA6: "ƒ", 0xA7: "♣", 0xA8: "♦",
    0xA9: "♥", 0xAA: "♠", 0xAB: "↔", 0xAC: "←", 0xAD: "↑", 0xAE: "→", 0xAF: "↓", 0xB0: "°",
    0xB1: "±", 0xB2: "″", 0xB3: "≥", 0xB4: "×", 0xB5: "∝", 0xB6: "∂", 0xB7: "•", 0xB8: "÷",
    0xB9: "≠", 0xBA: "≡", 0xBB: "≈", 0xBC: "…", 0xBD: "|", 0xBE: "—", 0xBF: "↵",
    0xC0: "ℵ", 0xC1: "ℑ", 0xC2: "ℜ", 0xC3: "℘", 0xC4: "⊗", 0xC5: "⊕", 0xC6: "∅", 0xC7: "∩",
    0xC8: "∪", 0xC9: "⊃", 0xCA: "⊇", 0xCB: "⊄", 0xCC: "⊂", 0xCD: "⊆", 0xCE: "∈", 0xCF: "∉",
    0xD0: "∠", 0xD1: "∇", 0xD2: "®", 0xD3: "©", 0xD4: "™", 0xD5: "∏", 0xD6: "√", 0xD7: "⋅",
    0xD8: "¬", 0xD9: "∧", 0xDA: "∨", 0xDB: "⇔", 0xDC: "⇐", 0xDD: "⇑", 0xDE: "⇒", 0xDF: "⇓",
    0xE0: "◊", 0xE1: "⟨", 0xE2: "®", 0xE3: "©", 0xE4: "™", 0xE5: "∑",
    0xE6: "(", 0xE7: "", 0xE8: "", 0xE9: "[", 0xEA: "", 0xEB: "", 0xEC: "{", 0xED: "", 0xEE: "", 0xEF: "",
    0xF1: "⟩", 0xF2: "∫", 0xF3: "∫", 0xF4: "", 0xF5: "",
    0xF6: ")", 0xF7: "", 0xF8: "", 0xF9: "]", 0xFA: "", 0xFB: "", 0xFC: "}", 0xFD: "", 0xFE: "",
}
_SYMBOL_TABLE = {0xF000 + code: char for code, char in _SYMBOL_FONT.items()}


def fix_symbol_font(text: str) -> str:
    """Replace Symbol-font private-use characters with the symbols they stand for."""
    return text.translate(_SYMBOL_TABLE)


_MATH_SYMBOLS = frozenset("=+−×÷√∠Δ≤≥≠±∝∴⊥≅∼≈πθαβγ∫∑′")
# A line holding only a short number: a piece of a fraction split over lines ("1" above "2").
_FRACTION_PIECE = re.compile(r"^[ \t]*\d{1,3}[ \t]*$", re.MULTILINE)


def needs_math_reading(text: str) -> bool:
    """Whether a page's text layer looks like it lost its math layout (call after fix_symbol_font)."""
    symbols = sum(1 for ch in text if ch in _MATH_SYMBOLS)
    return symbols >= 6 or len(_FRACTION_PIECE.findall(text)) >= 3


PAGE_PROMPT = r"""Transcribe this textbook page into plain text.
- Copy the text exactly, in reading order, keeping headings, numbering (Example 3, Theorem 6.1, EXERCISE 2.1, (i), (ii)) and paragraph breaks.
- Write every mathematical expression in LaTeX: inline as $...$ and displayed equations as $$...$$, e.g. $x^2 - 2x - 8$, $\frac{AD}{DB} = \frac{3}{2}$, $\angle ABC = 90^\circ$, $\triangle ABC \sim \triangle PQR$, $\sqrt{3}$.
- For each diagram or graph, write one line: [Figure: its label if any (e.g. Fig. 6.9) and what it shows, in a few words].
- Leave out running headers, page numbers, footers and watermarks.
- Do not solve, explain, summarise or add anything. Output only the transcription."""

# Long side of the page picture sent to the vision model: textbook print stays legible, with
# ~35% fewer image tokens than 1600 px (free tiers cap tokens per day).
PAGE_IMAGE_SIDE = 1280
# Seconds per page, including waits on per-minute rate limits; past it, the page is retried later.
PAGE_TIMEOUT = 300
_FENCE = re.compile(r"^```[a-z]*\n|\n```$")


def render_pages(path: Path, numbers: list[int]) -> dict[int, bytes]:
    """JPEG pictures of the given PDF pages (numbered from 1)."""
    import io

    import pypdfium2 as pdfium

    images: dict[int, bytes] = {}
    pdf = pdfium.PdfDocument(str(path))
    try:
        for number in numbers:
            page = pdf[number - 1]
            try:
                scale = PAGE_IMAGE_SIDE / max(page.get_size())
                buffer = io.BytesIO()
                page.render(scale=scale).to_pil().convert("RGB").save(buffer, "JPEG", quality=85)
                images[number] = buffer.getvalue()
            finally:
                page.close()
    finally:
        pdf.close()
    return images


def _visible_chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _accept(transcript: str, original: str) -> str | None:
    """The transcript, unless it is empty or much shorter than the page's own text (a refusal or
    a cut-off reply would otherwise replace real content)."""
    transcript = _FENCE.sub("", transcript.strip()).strip()
    if not transcript or _visible_chars(transcript) < 0.5 * _visible_chars(original):
        return None
    return transcript


async def _reserve(user_id: uuid.UUID) -> bool:
    async with SessionLocal() as session:
        try:
            await usage.reserve(session, user_id, "math_page")
        except AppError:
            return False
        await session.commit()
        return True


async def _refund(user_id: uuid.UUID) -> None:
    async with SessionLocal() as session:
        await usage.refund(session, user_id, "math_page")
        await session.commit()


@dataclass
class MathReading:
    texts: dict[int, str]  # page number -> transcription with LaTeX
    deferred: int = 0  # pages left for later because every vision provider was rate-limited


async def _saved_transcripts(document_id: uuid.UUID, numbers: list[int]) -> dict[int, str]:
    async with SessionLocal() as session:
        rows = await session.execute(
            select(PageTranscript.page, PageTranscript.text).where(
                PageTranscript.document_id == document_id, PageTranscript.page.in_(numbers)
            )
        )
        return dict(rows.all())


async def _save_transcript(document_id: uuid.UUID, number: int, text: str) -> None:
    async with SessionLocal() as session:
        await session.merge(PageTranscript(document_id=document_id, page=number, text=text))
        await session.commit()


async def page_transcripts(document_id: uuid.UUID) -> dict[int, str]:
    """Every saved transcript of a document, by page (used to caption its figures)."""
    async with SessionLocal() as session:
        rows = await session.execute(
            select(PageTranscript.page, PageTranscript.text).where(PageTranscript.document_id == document_id)
        )
        return dict(rows.all())


async def read_math_pages(
    path: Path, pages: list[tuple[int | None, str]], user_id: uuid.UUID, document_id: uuid.UUID, name: str = ""
) -> MathReading:
    """LaTeX transcriptions of the PDF pages whose math the text layer garbled.

    Pages read before (saved in page_transcripts) cost nothing. Pages it can't read keep their
    text-layer text: over the per-document or daily limit, reply rejected, or, counted in
    `deferred`, a passing problem: rate limits, timeouts, network or server errors (the caller
    schedules those for later).
    """
    settings = get_settings()
    name = name or path.name
    if not settings.math_vision_enabled or path.suffix.lower() != ".pdf":
        return MathReading({})
    originals = {n: text for n, text in pages if n is not None and needs_math_reading(text)}
    wanted = sorted(originals)[: settings.math_vision_max_pages]
    if not wanted:
        return MathReading({})
    texts = await _saved_transcripts(document_id, wanted)
    todo = [n for n in wanted if n not in texts]
    stats = {"saved before": len(texts), "read": 0, "over daily limit": 0, "rejected": 0, "failed": 0, "deferred": 0}
    if todo:
        images = await asyncio.to_thread(render_pages, path, todo)
        slots = asyncio.Semaphore(3)  # quick, without tripping the vision model's rate limit

        async def read(number: int) -> None:
            async with slots:
                if not await _reserve(user_id):
                    stats["over daily limit"] += 1
                    return
                try:
                    reply = await asyncio.wait_for(providers.vision().transcribe(images[number], PAGE_PROMPT), PAGE_TIMEOUT)
                except Exception as exc:
                    await _refund(user_id)
                    if is_retryable(exc):
                        logger.warning("Page %d of %r left for later: %s", number, name, type(exc).__name__)
                        stats["deferred"] += 1
                    else:
                        logger.warning("Could not read page %d of %r with the vision model", number, name, exc_info=True)
                        stats["failed"] += 1
                    return
                text = _accept(reply, originals[number])
                if text is None:
                    stats["rejected"] += 1
                    return
                stats["read"] += 1
                texts[number] = text
                await _save_transcript(document_id, number, text)  # kept even if a later page fails

        await asyncio.gather(*(read(n) for n in todo))
    logger.info(
        "%r: %d math page(s) of %d: %s", name, len(wanted), len(pages),
        ", ".join(f"{n} {what}" for what, n in stats.items() if n),
    )
    return MathReading(texts, stats["deferred"])
